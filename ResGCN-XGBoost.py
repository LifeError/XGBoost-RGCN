import sys
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph
import logging
import xgboost as xgb
# 全局参数
seed = 42
batch_size = 512
lr = 0.001
min_lr = 1e-7
num_epochs = 5000
test_epochs = 20
hidden_sizes = [512,512,512]
no_cuda = False  # 启用CUDA
patience = 10
l2_reg = 0.0001
accumulation_steps = 4
dropout_rate = 0.1  # 添加Dropout率
recall_weight = 2.0  # Recall的权重
fpr_weight = 1.0     # FPR的权重
threshold = 0.7 # 阈值参数
thresholds_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]  # 要测试的阈值列表


# 配置日志记录 - 同时输出到控制台和文件
def setup_logging():
    # 清除现有配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 设置新配置（无颜色）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # 控制台输出（无颜色）
            logging.FileHandler('ResGCN-XGBoost_logging_results.txt')  # 文件输出
        ]
    )

# 立即设置日志配置
setup_logging()

# 打印关键参数到日志开头
logging.info("=" * 60)
logging.info("实验参数配置:")
logging.info(f"hidden_sizes: {hidden_sizes}")
logging.info(f"dropout_rate: {dropout_rate}")
logging.info(f"threshold: {threshold}")
logging.info("=" * 60)

# 测试日志
logging.info("日志系统已成功配置")

# 计算正负样本权重
def calculate_class_weights(labels):
    num_pos = np.sum(labels)
    num_neg = len(labels) - num_pos
    total = num_pos + num_neg
    weight_pos = total / (2 * num_pos)
    weight_neg = total / (2 * num_neg)
    return torch.tensor([weight_neg, weight_pos], dtype=torch.float)


# GPU内存监控
def print_gpu_memory(device):
    if torch.cuda.is_available():
        logging.info(f"GPU内存使用情况 - 分配: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB, "
              f"缓存: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


# 设置随机种子，保证实验可复现性
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 矩阵归一化函数，用于特征矩阵预处理
def normalize(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


# 基于ResGCN的图卷积网络模型
class ResGCN(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features):
        super(ResGCN, self).__init__()
        self.input_layer = GCNConv(in_features, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()  # 添加Dropout层列表
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(GCNConv(hidden_sizes[i], hidden_sizes[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout_rate))  # 添加Dropout层
        self.output_layer = nn.Linear(hidden_sizes[-1], out_features)

    def forward(self, x, edge_index):
        identity = x
        x = F.relu(self.input_layer(x, edge_index))
        for i, layer in enumerate(self.hidden_layers):
            out = F.relu(layer(x, edge_index))
            # 残差连接
            if out.shape == x.shape:
                out = out + x  # 修改为非原地操作
            x = out
            x = self.dropout_layers[i](x)  # 应用Dropout
        return self.output_layer(x)


# 模型评估函数，计算各种性能指标
def evaluate_model(test_nodes, all_labels, model, features, edge_index, batch_size, device, threshold):
    model.eval()
    all_preds, all_targets, all_scores = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            batch_nodes = test_nodes[i:i + batch_size]
            batch_node_idx = torch.LongTensor(batch_nodes).to(device)

            # 提取当前批次节点对应的子图
            subgraph_result = subgraph(
                batch_node_idx,
                edge_index,
                relabel_nodes=True,
                num_nodes=features.size(0)
            )

            # 根据subgraph函数的返回值数量进行处理
            if len(subgraph_result) == 3:
                batch_edge_index, _, _ = subgraph_result
            else:  # 处理返回两个值的情况
                batch_edge_index, _ = subgraph_result

            index = batch_node_idx
            batch_label = all_labels[batch_nodes]
            label = torch.LongTensor(batch_label).to(device)
            out = model(features[index], batch_edge_index)
            probs = torch.softmax(out, dim=1)
            scores = probs[:, 1].cpu().numpy()
            preds = (scores > threshold).astype(int)  # 使用阈值进行预测
            all_preds.extend(preds)
            all_targets.extend(batch_label)
            all_scores.extend(scores)

    f1 = f1_score(all_targets, all_preds, average="macro")
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=1)
    recall = recall_score(all_targets, all_preds, average="macro")
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(all_targets, all_scores)

    return {"f1": f1, "precision": precision, "recall": recall, "fpr": fpr, "auc": auc}


# 数据加载
prefix = 'work/'
data = loadmat(prefix + 'YelpChi.mat')

# 查看数据结构
logging.info(f"数据文件中的键：{list(data.keys())}")

labels = data['label'].flatten()
feat_data = data['features'].todense().A

# 划分数据集
# 首先划分出20%的测试集
index = list(range(len(labels)))
idx_train_val, idx_test = train_test_split(index, stratify=labels, test_size=0.2, random_state=seed)
# 然后将剩余的80%划分为70%训练集和10%验证集
idx_train, idx_val = train_test_split(idx_train_val, stratify=labels[idx_train_val], test_size=0.125, random_state=seed)

# Step 1: 训练XGBoost并输出预测概率（正负类别的概率）
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=seed)
xgb_model.fit(feat_data[idx_train], labels[idx_train])

# 获取每个节点属于各类别的概率（shape: N x 2）
xgb_probs = xgb_model.predict_proba(feat_data)

# Step 2: 将原始特征与XGBoost概率拼接
# feat_data shape: (45954, 32) ; xgb_probs shape: (45954, 2)
feat_data_aug = np.hstack([feat_data, xgb_probs])

# 归一化并转成torch
feat_data = normalize(feat_data_aug)

# 查找边信息
try:
    possible_edge_keys = ['homo', 'net_rur', 'net_rtr', 'net_rsr']
    edge_key = None
    edge_info = []
    for i, key in enumerate(possible_edge_keys):
        if key in data:
            edge = data[key]
            if sp.issparse(edge) and edge.nnz > 0:
                edge = edge.tocoo()
                edge_index = torch.stack([
                    torch.tensor(edge.row, dtype=torch.long),
                    torch.tensor(edge.col, dtype=torch.long)
                ], dim=0)
                edge_info.append(edge_index)
            elif isinstance(edge, np.ndarray) and edge.size > 0:
                if edge.ndim == 2 and edge.shape[1] == 2:
                    edge_index = torch.tensor(edge, dtype=torch.long).t()
                else:
                    rows, cols = np.nonzero(edge)
                    edge_index = torch.stack([
                        torch.tensor(rows, dtype=torch.long),
                        torch.tensor(cols, dtype=torch.long)
                    ], dim=0)
                edge_info.append(edge_index)

    if not edge_info:
        raise KeyError("找不到合适的边信息，请检查数据文件结构")

    edge_index = torch.cat(edge_info, dim=1)

    logging.info(f"成功合并所有边信息")
    logging.info(f"边的数量: {edge_index.shape[1]}")

except Exception as e:
    logging.info(f"处理边信息时出错: {e}")
    edge_index = torch.empty(2, 0, dtype=torch.long)

# 设置计算设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
logging.info(f"使用设备: {device}")

# 将数据移动到设备上
features = torch.FloatTensor(feat_data).to(device)
edge_index = edge_index.to(device)

# 计算类别权重，用于处理类别不平衡问题
class_weights = calculate_class_weights(labels[idx_train]).to(device)

# 初始化模型和优化器
model = ResGCN(in_features=feat_data.shape[1], hidden_sizes=hidden_sizes, out_features=2).to(
    device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

# 早停设置
best_metric = float('-inf')
no_improve = 0
best_model_state = None

# 打印初始GPU内存使用情况
if device.type == "cuda":
    scaler = GradScaler(enabled=True)

# 打印初始GPU内存
if device.type == "cuda":
    print_gpu_memory(device)

# 训练循环并使用早停策略
for epoch in range(num_epochs):
    model.train()
    random.shuffle(idx_train)
    total_loss = 0
    optimizer.zero_grad()

    for i in range(0, len(idx_train), batch_size):
        batch_nodes = idx_train[i:i + batch_size]
        batch_node_idx = torch.LongTensor(batch_nodes).to(device)

        # 提取当前批次节点对应的子图
        subgraph_result = subgraph(
            batch_node_idx,
            edge_index,
            relabel_nodes=True,
            num_nodes=features.size(0)
        )

        # 根据subgraph函数的返回值数量进行处理
        if len(subgraph_result) == 3:
            batch_edge_index, _, _ = subgraph_result
        else:  # 处理返回两个值的情况
            batch_edge_index, _ = subgraph_result

        batch_label = labels[batch_nodes]
        index = batch_node_idx
        label = torch.LongTensor(batch_label).to(device)

        # 混合精度训练
        if device.type == "cuda":
            with autocast(device_type='cuda'):
                out = model(features[index], batch_edge_index)
                loss = criterion(out, label) / accumulation_steps
        else:
            out = model(features[index], batch_edge_index)
            loss = criterion(out, label) / accumulation_steps

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

    scheduler.step()

    # 每100个epoch打印一次GPU内存
    if epoch % 100 == 0 and device.type == "cuda":
        print_gpu_memory(device)

    if epoch % test_epochs == 0:
        # 使用验证集进行评估
        metrics = evaluate_model(idx_val, labels, model, features, edge_index, batch_size, device, threshold)
        current_metric = recall_weight * metrics['recall'] - fpr_weight * metrics['fpr']
        logging.info(
            f"Epoch {epoch} | Loss: {total_loss:.4f} | F1: {metrics['f1']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
            f"FPR: {metrics['fpr']:.4f} | AUC: {metrics['auc']:.4f} | "
            f"Early Stopping Metric: {current_metric:.4f}"
        )

        # 早停检查
        if current_metric > best_metric:
            best_metric = current_metric
            no_improve = 0
            best_model_state = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

# 训练完成后，加载最佳模型并测试
logging.info("Training complete, loading best model...")
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# 最终评估，使用测试集
final_metrics = evaluate_model(idx_test, labels, model, features, edge_index, batch_size, device, threshold)
logging.info("Final Evaluation after training:")
logging.info(f"F1: {final_metrics['f1']:.4f}")
logging.info(f"Precision: {final_metrics['precision']:.4f}")
logging.info(f"Recall: {final_metrics['recall']:.4f}")
logging.info(f"FPR: {final_metrics['fpr']:.4f}")
logging.info(f"AUC: {final_metrics['auc']:.4f}")

# 打印不同阈值下的测试结果
logging.info("\n不同阈值下的测试集结果:")
logging.info("=" * 60)
logging.info("阈值\tF1\t\tPrecision\t\tRecall\t\tFPR\t\tAUC")
logging.info("-" * 60)

for thresh in thresholds_to_test:
    metrics = evaluate_model(idx_test, labels, model, features, edge_index, batch_size, device, thresh)
    logging.info(f"{thresh:.2f}\t\t{metrics['f1']:.4f}\t{metrics['precision']:.4f}\t\t\t{metrics['recall']:.4f}\t\t{metrics['fpr']:.4f}\t\t{metrics['auc']:.4f}")

logging.info("=" * 60)

# 打印最终GPU内存
if device.type == "cuda":
    print_gpu_memory(device)