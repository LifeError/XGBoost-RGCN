import sys
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import xgboost as xgb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import subgraph
import logging
import pandas as pd
import sklearn

# 全局参数
seed = 42
batch_size = 1024
lr = 0.0001
min_lr = 1e-7
num_epochs = 5000
hidden_sizes = [128, 128, 128]
no_cuda = False
patience = 10  # 早停耐心值（基于训练损失）
l2_reg = 0.001
accumulation_steps = 2
dropout_rate = 0.2
recall_weight = 2.0
fpr_weight = 1.0
threshold = 0.7
thresholds_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]

# 仅训练集+测试集的时间戳划分（按文献逻辑）
TRAIN_TS_RANGE = (1, 30)  # 训练集：1~30
TEST_TS_RANGE = (31, 49)  # 测试集：31~49


# 配置日志
def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('RGCN-XGBoost_Elliptic_FinalOnlyEval.txt')
        ]
    )


setup_logging()

# 打印核心配置
logging.info("=" * 60)
logging.info("配置：仅训练集+测试集 | 仅训练结束后评估测试集 | 移除SMOTE（用类别权重处理不平衡）")
logging.info(f"时间戳划分：训练集{TRAIN_TS_RANGE} | 测试集{TEST_TS_RANGE}")
logging.info(f"hidden_sizes: {hidden_sizes} | dropout_rate: {dropout_rate} | threshold: {threshold}")
logging.info("=" * 60)


# 计算类别权重（核心：用损失权重处理不平衡，替代SMOTE）
def calculate_class_weights(labels):
    num_pos = np.sum(labels)
    num_neg = len(labels) - num_pos
    total = num_pos + num_neg
    # 平衡权重：少数类（欺诈）权重更高
    weight_pos = total / (2 * num_pos) if num_pos > 0 else 1.0
    weight_neg = total / (2 * num_neg) if num_neg > 0 else 1.0
    return torch.tensor([weight_neg, weight_pos], dtype=torch.float)


# GPU内存监控
def print_gpu_memory(device):
    if torch.cuda.is_available():
        logging.info(
            f"GPU内存 - 分配: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB, 缓存: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


# 设置随机种子
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 特征归一化
def normalize(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


# RGCN模型（不变）
class RGCN(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features, num_relations):
        super(RGCN, self).__init__()
        self.input_layer = RGCNConv(in_features, hidden_sizes[0], num_relations)
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(RGCNConv(hidden_sizes[i], hidden_sizes[i + 1], num_relations))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        self.output_layer = nn.Linear(hidden_sizes[-1], out_features)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.input_layer(x, edge_index, edge_type))
        for i, layer in enumerate(self.hidden_layers):
            x = F.relu(layer(x, edge_index, edge_type))
            x = self.dropout_layers[i](x)
        return self.output_layer(x)


# 评估函数：输出全量指标（每个类+宏观+加权+整体）
def evaluate_model(test_nodes, all_labels, model, features, edge_index, edge_type, batch_size, device, threshold):
    model.eval()
    all_preds, all_targets, all_scores = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            batch_nodes = test_nodes[i:i + batch_size]
            batch_node_idx = torch.LongTensor(batch_nodes).to(device)

            # 提取批次子图
            subgraph_result = subgraph(
                batch_node_idx, edge_index, edge_attr=edge_type, relabel_nodes=True, num_nodes=features.size(0)
            )
            if len(subgraph_result) == 3:
                batch_edge_index, _, batch_edge_type = subgraph_result
            else:
                batch_edge_index, edge_mask = subgraph_result
                batch_edge_type = edge_type[edge_mask]

            batch_label = all_labels[batch_nodes]
            out = model(features[batch_node_idx], batch_edge_index, batch_edge_type)
            probs = torch.softmax(out, dim=1)
            scores = probs[:, 1].cpu().numpy()
            preds = (scores > threshold).astype(int)

            all_preds.extend(preds)
            all_targets.extend(batch_label)
            all_scores.extend(scores)

    # ========== 计算全量指标 ==========
    # 1. 整体准确率
    acc = sklearn.metrics.accuracy_score(all_targets, all_preds)

    # 2. 每个类（0=合法，1=欺诈）的Precision/Recall/F1
    prec_per_class = precision_score(all_targets, all_preds, average=None, labels=[0, 1], zero_division=1)
    recall_per_class = recall_score(all_targets, all_preds, average=None, labels=[0, 1], zero_division=1)
    f1_per_class = f1_score(all_targets, all_preds, average=None, labels=[0, 1], zero_division=1)

    # 3. 宏观平均（macro）：平等加权每个类
    prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=1)
    recall_macro = recall_score(all_targets, all_preds, average="macro", zero_division=1)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=1)

    # 4. 加权平均（weighted）：按样本数加权每个类
    prec_weighted = precision_score(all_targets, all_preds, average="weighted", zero_division=1)
    recall_weighted = recall_score(all_targets, all_preds, average="weighted", zero_division=1)
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=1)

    # 5. 其他辅助指标（FPR/TPR/AUC）
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(all_targets, all_scores) if len(set(all_targets)) > 1 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

    # 封装所有指标
    metrics = {
        # 整体准确率
        "acc": acc,
        # 每个类的指标（0=合法，1=欺诈）
        "prec_class_0": prec_per_class[0],  # 合法类精确率
        "prec_class_1": prec_per_class[1],  # 欺诈类精确率
        "recall_class_0": recall_per_class[0],  # 合法类召回率
        "recall_class_1": recall_per_class[1],  # 欺诈类召回率
        "f1_class_0": f1_per_class[0],  # 合法类F1
        "f1_class_1": f1_per_class[1],  # 欺诈类F1
        # 宏观平均
        "prec_macro": prec_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        # 加权平均
        "prec_weighted": prec_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        # 辅助指标
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc
    }
    return metrics


# 加载数据（仅保留训练/测试集相关）
def load_elliptic_data():
    nodes_df = pd.read_csv("elliptic_txs_features.csv", header=None)
    classes_df = pd.read_csv("elliptic_txs_classes.csv")

    # 列映射：0=txId，1=timestep(1~49)，2~165=特征
    node_ids = nodes_df.iloc[:, 0].values
    timesteps = nodes_df.iloc[:, 1].values
    feat_data = nodes_df.iloc[:, 2:].values

    # 标签映射：1=欺诈，2=合法，unknown=-1
    class_map = {"1": 1, "2": 0, "unknown": -1}
    classes_df["label"] = classes_df["class"].map(class_map)
    label_dict = dict(zip(classes_df["txId"], classes_df["label"]))
    labels = np.array([label_dict.get(node_id, -1) for node_id in node_ids])

    # 过滤：仅保留有标签的节点
    valid_mask = labels != -1
    valid_node_ids = node_ids[valid_mask]
    valid_feats = feat_data[valid_mask]
    valid_labels = labels[valid_mask]
    valid_timesteps = timesteps[valid_mask]

    # 节点ID→索引/时间戳映射
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(valid_node_ids)}
    node_id_to_ts = dict(zip(valid_node_ids, valid_timesteps))

    # 加载边并过滤：仅保留src_ts ≤ dst_ts（时序约束）
    edges_df = pd.read_csv("elliptic_txs_edgelist.csv")
    valid_edges = []
    for src, dst in edges_df[["txId1", "txId2"]].values:
        if src in node_id_to_idx and dst in node_id_to_idx:
            src_ts = node_id_to_ts[src]
            dst_ts = node_id_to_ts[dst]
            if src_ts <= dst_ts:
                valid_edges.append([node_id_to_idx[src], node_id_to_idx[dst]])
    valid_edges = np.array(valid_edges)

    # 打印数据统计
    logging.info(f"Elliptic数据集加载完成：")
    logging.info(f"  有效节点数：{len(valid_node_ids)}（时间戳范围：{np.min(valid_timesteps)}~{np.max(valid_timesteps)}）")
    logging.info(f"  欺诈节点：{np.sum(valid_labels == 1)} | 合法节点：{np.sum(valid_labels == 0)}")
    logging.info(f"  有效边数：{len(valid_edges)} | 特征维度：{valid_feats.shape[1]}")

    return valid_feats, valid_edges, valid_labels, valid_timesteps


# 预处理数据：仅划分训练/测试集（移除SMOTE）
def preprocess_elliptic_data(feat_data, edges, labels, timesteps):
    # 构建PyG边索引
    src = edges[:, 0]
    dst = edges[:, 1]
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_index = torch.LongTensor(edge_index)
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)

    # 仅划分训练/测试集（移除验证集）
    train_mask = (timesteps >= TRAIN_TS_RANGE[0]) & (timesteps <= TRAIN_TS_RANGE[1])
    test_mask = (timesteps >= TEST_TS_RANGE[0]) & (timesteps <= TEST_TS_RANGE[1])

    idx_train = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    # 移除SMOTE：直接使用原始特征（无样本替换）
    feat_data_norm = normalize(feat_data)  # 仅归一化，无过采样
    features = torch.FloatTensor(feat_data_norm)

    # 打印划分统计
    logging.info(f"数据预处理完成（仅训练/测试集，移除SMOTE）：")
    logging.info(
        f"  训练集（{TRAIN_TS_RANGE[0]}~{TRAIN_TS_RANGE[1]}）：{len(idx_train)}节点（欺诈：{np.sum(labels[idx_train] == 1)}）")
    logging.info(
        f"  测试集（{TEST_TS_RANGE[0]}~{TEST_TS_RANGE[1]}）：{len(idx_test)}节点（欺诈：{np.sum(labels[idx_test] == 1)}）")

    return features, edge_index, edge_type, labels, idx_train, idx_test


# 加载数据
feat_data, edges, labels, timesteps = load_elliptic_data()
features, edge_index, edge_type, labels, idx_train, idx_test = preprocess_elliptic_data(
    feat_data, edges, labels, timesteps
)

# XGBoost特征增强（仅训练集训练，分批次预测）
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=seed)
xgb_model.fit(feat_data[idx_train], labels[idx_train])


def batch_predict_proba(model, X, batch_size=1024):
    probs = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i + batch_size]
        batch_probs = model.predict_proba(batch_X)
        probs.append(batch_probs)
    return np.vstack(probs)


xgb_probs = batch_predict_proba(xgb_model, feat_data)
feat_data_aug = np.hstack([feat_data, xgb_probs])
feat_data = normalize(feat_data_aug)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
logging.info(f"使用设备: {device}")

# 数据移至设备
features = torch.FloatTensor(feat_data).to(device)
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)

# 类别权重（基于原始训练集标签计算）
class_weights = calculate_class_weights(labels[idx_train]).to(device)
logging.info(f"类别权重（0=合法，1=欺诈）：{class_weights.cpu().numpy()}")

# 模型初始化
model = RGCN(in_features=feat_data.shape[1], hidden_sizes=hidden_sizes, out_features=2, num_relations=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
criterion = nn.CrossEntropyLoss(weight=class_weights)  # 用类别权重平衡损失
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
scaler = GradScaler(enabled=(device.type == "cuda"))  # 混合精度训练

# 早停设置（基于训练损失，无中间测试集评估）
best_loss = float('inf')  # 初始化最佳损失为无穷大
no_improve = 0
best_model_state = None

# 打印初始GPU内存
if device.type == "cuda":
    print_gpu_memory(device)

# 训练循环（仅训练，无中间轮次评估）
logging.info("开始训练模型（无中间轮次评估，仅训练结束后评估测试集）...")
for epoch in range(num_epochs):
    model.train()
    random.shuffle(idx_train)
    total_loss = 0
    optimizer.zero_grad()

    # 批次训练
    for i in range(0, len(idx_train), batch_size):
        batch_nodes = idx_train[i:i + batch_size]
        batch_node_idx = torch.LongTensor(batch_nodes).to(device)

        # 提取批次子图
        subgraph_result = subgraph(
            batch_node_idx, edge_index, edge_attr=edge_type, relabel_nodes=True, num_nodes=features.size(0)
        )
        if len(subgraph_result) == 3:
            batch_edge_index, _, batch_edge_type = subgraph_result
        else:
            batch_edge_index, edge_mask = subgraph_result
            batch_edge_type = edge_type[edge_mask]

        batch_label = labels[batch_nodes]
        label = torch.LongTensor(batch_label).to(device)

        # 混合精度前向传播
        if device.type == "cuda":
            with autocast(device_type='cuda'):
                out = model(features[batch_node_idx], batch_edge_index, batch_edge_type)
                loss = criterion(out, label) / accumulation_steps
        else:
            out = model(features[batch_node_idx], batch_edge_index, batch_edge_type)
            loss = criterion(out, label) / accumulation_steps

        # 反向传播（梯度累积）
        if device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i // batch_size + 1) % accumulation_steps == 0:
            if device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # 学习率调度
    scheduler.step()

    # 每100轮打印GPU内存和训练损失（仅监控，不评估测试集）
    if epoch % 100 == 0:
        logging.info(f"Epoch {epoch} | 训练损失: {total_loss:.4f}")
        if device.type == "cuda":
            print_gpu_memory(device)

    # 早停判断（基于训练损失）
    if total_loss < best_loss:
        best_loss = total_loss
        no_improve = 0
        best_model_state = model.state_dict()
    else:
        no_improve += 1
        if no_improve >= patience:
            logging.info(f"早停触发：Epoch {epoch}（训练损失连续{patience}轮无下降）")
            break

# 训练完成后：加载最佳模型 + 仅一次评估测试集
logging.info("=" * 60)
logging.info("训练完成，加载最佳模型并评估测试集（仅最后一次评估）")
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# 最终测试集评估（全量指标）
final_metrics = evaluate_model(idx_test, labels, model, features, edge_index, edge_type, batch_size, device, threshold)
logging.info("=" * 60)
logging.info("最终测试集全量评估结果：")
logging.info(f"1. 整体准确率 (Acc): {final_metrics['acc']:.4f}")
logging.info("-" * 60)
logging.info("2. 单类别指标（0=合法，1=欺诈）：")
logging.info(
    f"   合法类 - 精确率: {final_metrics['prec_class_0']:.4f} | 召回率: {final_metrics['recall_class_0']:.4f} | F1: {final_metrics['f1_class_0']:.4f}")
logging.info(
    f"   欺诈类 - 精确率: {final_metrics['prec_class_1']:.4f} | 召回率: {final_metrics['recall_class_1']:.4f} | F1: {final_metrics['f1_class_1']:.4f}")
logging.info("-" * 60)
logging.info("3. 宏观平均指标（平等加权）：")
logging.info(
    f"   精确率: {final_metrics['prec_macro']:.4f} | 召回率: {final_metrics['recall_macro']:.4f} | F1: {final_metrics['f1_macro']:.4f}")
logging.info("-" * 60)
logging.info("4. 加权平均指标（按样本数加权）：")
logging.info(
    f"   精确率: {final_metrics['prec_weighted']:.4f} | 召回率: {final_metrics['recall_weighted']:.4f} | F1: {final_metrics['f1_weighted']:.4f}")
logging.info("-" * 60)
logging.info("5. 辅助指标：")
logging.info(f"   FPR: {final_metrics['fpr']:.4f} | TPR: {final_metrics['tpr']:.4f} | AUC: {final_metrics['auc']:.4f}")
logging.info("=" * 60)

# 不同阈值下的测试集最终评估
logging.info("\n不同阈值下的测试集最终评估结果：")
logging.info("=" * 90)
logging.info("阈值\tAcc\t合法类F1\t欺诈类F1\t宏观F1\t加权F1\tAUC")
logging.info("-" * 90)
for thresh in thresholds_to_test:
    metrics = evaluate_model(idx_test, labels, model, features, edge_index, edge_type, batch_size, device, thresh)
    logging.info(
        f"{thresh:.2f}\t\t{metrics['acc']:.4f}\t\t{metrics['f1_class_0']:.4f}\t\t{metrics['f1_class_1']:.4f}\t\t{metrics['f1_macro']:.4f}\t\t{metrics['f1_weighted']:.4f}\t\t{metrics['auc']:.4f}"
    )
logging.info("=" * 90)

# 打印最终GPU内存
if device.type == "cuda":
    print_gpu_memory(device)
logging.info("所有评估完成，结果已保存至日志文件")