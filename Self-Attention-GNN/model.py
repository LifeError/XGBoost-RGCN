import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import subgraph, from_scipy_sparse_matrix
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import os
import copy

# ===================== 1. 全局配置（优化抗过拟合参数） =====================
DATASET_PATH = r'E:\Python\projectes\lunwen\work\YelpChi.mat'
TRAIN_RATIO = 0.7  # 减少训练集比例，留出验证集
VAL_RATIO = 0.1  # 验证集比例
TEST_RATIO = 0.2  # 测试集比例
RANDOM_SEED = 42

# 模型参数（降低复杂度+增强正则化）
HIDDEN_SIZES = [128, 128]  # 减少隐藏层数量和维度
DROPOUT_RATE = 0.5  # 增大Dropout率
THRESHOLD = 0.7
BATCH_SIZE = 1024
MAX_EPOCHS = 1000  # 减少最大训练轮数
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001  # 增大权重衰减
PATIENCE = 10  # 早停耐心值（20轮无提升则停止）

# 日志配置
LOG_FILE_PATH = "train_eval_log_anti_overfit.txt"
EDGE_KEYS = ['homo', 'net_rur', 'net_rtr', 'net_rsr']


# ===================== 2. 日志配置 =====================
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# ===================== 3. 通用评估函数（新增验证集支持） =====================
def calculate_metrics(model, features, edge_index, edge_type, cases, labels, batch_size, device):
    model.eval()
    total_loss = 0.0
    f1 = 0.0
    acc = 0.0
    precision = 0.0
    recall = 0.0
    gnn_list = []
    all_labels = []

    pos_num = np.sum(labels == 1)
    neg_num = np.sum(labels == 0)
    pos_weight = torch.tensor(neg_num / pos_num).to(device) if pos_num > 0 else torch.tensor(1.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))

    batch_num = int(len(cases) / batch_size) + 1

    with torch.no_grad():
        for iteration in range(batch_num):
            i_start = iteration * batch_size
            i_end = min((iteration + 1) * batch_size, len(cases))
            batch_nodes = cases[i_start:i_end]
            batch_label = labels[i_start:i_end]

            index = torch.LongTensor(batch_nodes).to(device)
            batch_label_tensor = torch.LongTensor(batch_label).to(device)

            batch_edge_index, batch_edge_type = get_subgraph_batch(index, edge_index, edge_type)
            batch_edge_index = batch_edge_index.to(device)

            out = model(features.index_select(0, index), batch_edge_index)
            loss = criterion(out, batch_label_tensor)
            total_loss += loss.item()

            pos_score = F.softmax(out, dim=1)
            prob = pos_score.data.cpu().numpy().argmax(axis=1)
            pos_prob = pos_score.data.cpu().numpy()[:, 1]

            f1 += f1_score(batch_label, prob, average="macro", zero_division=1)
            acc += accuracy_score(batch_label, prob)
            precision += precision_score(batch_label, prob, average="macro", zero_division=1)
            recall += recall_score(batch_label, prob, average="macro", zero_division=1)

            gnn_list.extend(pos_prob.tolist())
            all_labels.extend(batch_label)

    avg_loss = total_loss / batch_num
    avg_f1 = f1 / batch_num
    avg_acc = acc / batch_num
    avg_precision = precision / batch_num
    avg_recall = recall / batch_num
    auc = roc_auc_score(all_labels, np.array(gnn_list)) if len(set(all_labels)) > 1 else 0.0

    model.train()
    return avg_loss, avg_f1, avg_acc, avg_precision, avg_recall, auc


# ===================== 4. YelpChi数据集加载函数（新增验证集划分） =====================
def load_aml_dataset():
    """
    加载数据集并划分训练/验证/测试集（7:1:2）
    """
    logger.info("开始加载YelpChi.mat数据集（合并所有边键）...")

    mat_data = loadmat(DATASET_PATH)
    logger.info(f"YelpChi.mat包含的键：{list(mat_data.keys())}")

    # 解析特征
    if 'features' in mat_data:
        features_sparse = mat_data['features']
    elif 'X' in mat_data:
        features_sparse = mat_data['X']
    else:
        raise ValueError("未找到特征键")
    features = torch.tensor(features_sparse.todense(), dtype=torch.float32)
    num_nodes = features.shape[0]
    logger.info(f"节点特征维度：{features.shape}")

    # 解析标签
    if 'label' in mat_data:
        labels = mat_data['label'].squeeze()
    elif 'y' in mat_data:
        labels = mat_data['y'].squeeze()
    else:
        raise ValueError("未找到标签键")
    labels = torch.tensor(labels, dtype=torch.long)
    logger.info(f"总节点数：{len(labels)}，欺诈节点数：{torch.sum(labels == 1).item()}")

    # 合并所有边
    all_edge_index = []
    all_edge_type = []
    edge_type_mapping = {}
    for edge_type_id, edge_key in enumerate(EDGE_KEYS):
        if edge_key in mat_data:
            adj_sparse = mat_data[edge_key]
            edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
            edge_type = torch.full((edge_index.shape[1],), edge_type_id, dtype=torch.long)

            edge_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, edge_mask]
            edge_type = edge_type[edge_mask]

            all_edge_index.append(edge_index)
            all_edge_type.append(edge_type)
            edge_type_mapping[edge_type_id] = edge_key
            logger.info(f"加载边键 {edge_key}：边数={edge_index.shape[1]}")
        else:
            logger.warning(f"边键 {edge_key} 不存在，跳过")

    if not all_edge_index:
        raise ValueError("未找到任何边键")

    edge_index = torch.cat(all_edge_index, dim=1)
    edge_type = torch.cat(all_edge_type, dim=0)

    # 兼容低版本的去重逻辑
    logger.info(f"合并后未去重的边数：{edge_index.shape[1]}")
    edge_index_t = edge_index.t().numpy()
    edge_index_unique, unique_indices = np.unique(edge_index_t, axis=0, return_index=True)
    unique_edges = torch.tensor(edge_index_unique).t()
    edge_type = edge_type[torch.tensor(unique_indices)]
    edge_index = unique_edges
    logger.info(f"合并所有边后（去重）：总边数={edge_index.shape[1]}")

    # 划分训练/验证/测试集（7:1:2）
    all_nodes = np.arange(num_nodes)
    # 第一步：划分训练+验证 和 测试
    train_val_nodes, test_nodes, train_val_labels, test_labels = train_test_split(
        all_nodes, labels.numpy(),
        train_size=TRAIN_RATIO + VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels.numpy()
    )
    # 第二步：划分训练 和 验证
    train_nodes, val_nodes, train_labels, val_labels = train_test_split(
        train_val_nodes, train_val_labels,
        train_size=TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_SEED,
        stratify=train_val_labels
    )

    logger.info(f"训练集：{len(train_nodes)}节点，验证集：{len(val_nodes)}节点，测试集：{len(test_nodes)}节点")
    logger.info("数据集加载完成！")

    return features, edge_index, edge_type, train_nodes, train_labels, val_nodes, val_labels, test_nodes, test_labels


# ===================== 5. 改进的Self-Attention GNN（添加BatchNorm+增强正则化） =====================
class SelfAttentionGNN(nn.Module):
    def __init__(self, in_channels, hidden_sizes=[128, 128], out_channels=2, dropout=0.5):
        super(SelfAttentionGNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_sizes = hidden_sizes
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)

        # GCN特征提取层（添加BatchNorm）
        self.gcn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()  # 批次归一化层
        self.gcn_layers.append(nn.Linear(in_channels, hidden_sizes[0]))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # 新增BatchNorm
        for i in range(1, len(hidden_sizes)):
            self.gcn_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[i]))  # 新增BatchNorm

        # 自注意力层
        self.attention_weight = nn.Linear(hidden_sizes[-1], hidden_sizes[-1])
        self.attention_score = nn.Linear(2 * hidden_sizes[-1], 1)

        # 输出层
        self.out_layer = nn.Linear(hidden_sizes[-1], out_channels)

    def gcn_forward(self, x):
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x)
            x = self.bn_layers[i](x)  # 批次归一化
            x = F.relu(x)
            x = self.dropout(x)  # Dropout增强
        return x

    def self_attention_agg(self, x, edge_index):
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]

        x_proj = self.attention_weight(x)
        h_i = x_proj[src_nodes]
        h_j = x_proj[tgt_nodes]
        h_concat = torch.cat([h_i, h_j], dim=1)
        attn_scores = F.leaky_relu(self.attention_score(h_concat))
        attn_scores = F.softmax(attn_scores, dim=0)

        agg_feat = torch.zeros_like(x_proj)
        agg_feat.index_add_(0, tgt_nodes, h_j * attn_scores)

        return agg_feat

    def forward(self, x, edge_index):
        gcn_feat = self.gcn_forward(x)
        attn_feat = self.self_attention_agg(gcn_feat, edge_index)
        fusion_feat = gcn_feat + attn_feat
        out = self.out_layer(fusion_feat)

        return out


# ===================== 6. 子图提取函数 =====================
def get_subgraph_batch(batch_nodes, edge_index, edge_type):
    subgraph_result = subgraph(
        batch_nodes,
        edge_index,
        edge_attr=edge_type,
        relabel_nodes=True,
        num_nodes=edge_index.max().item() + 1
    )
    if len(subgraph_result) == 3:
        batch_edge_index, _, batch_edge_type = subgraph_result
    else:
        batch_edge_index, edge_mask = subgraph_result
        batch_edge_type = edge_type[edge_mask]
    return batch_edge_index, batch_edge_type


# ===================== 7. 训练函数（添加早停+学习率衰减+验证集监控） =====================
def train_model(model, features, edge_index, edge_type, train_cases, train_labels, val_cases, val_labels, device):
    model.train()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 学习率衰减：验证集loss不下降时降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    pos_num = np.sum(train_labels == 1)
    neg_num = np.sum(train_labels == 0)
    pos_weight = torch.tensor(neg_num / pos_num).to(device) if pos_num > 0 else torch.tensor(1.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))

    train_batch_num = int(len(train_cases) / BATCH_SIZE) + 1
    logger.info("=" * 80)
    logger.info("开始训练Self-Attention GNN模型（抗过拟合版本）")
    logger.info(
        f"训练参数：hidden_sizes={HIDDEN_SIZES}, dropout={DROPOUT_RATE}, batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}")
    logger.info("=" * 80)

    # 早停相关变量
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        # 单epoch训练
        for iteration in range(train_batch_num):
            i_start = iteration * BATCH_SIZE
            i_end = min((iteration + 1) * BATCH_SIZE, len(train_cases))
            batch_nodes = train_cases[i_start:i_end]
            batch_label = train_labels[i_start:i_end]

            index = torch.LongTensor(batch_nodes).to(device)
            batch_label = torch.LongTensor(batch_label).to(device)

            batch_edge_index, _ = get_subgraph_batch(index, edge_index, edge_type)
            batch_edge_index = batch_edge_index.to(device)

            optimizer.zero_grad()
            out = model(features.index_select(0, index), batch_edge_index)
            loss = criterion(out, batch_label)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 每10个epoch计算训练集和验证集指标
        if (epoch + 1) % 10 == 0:
            avg_train_loss = epoch_loss / train_batch_num
            # 计算训练集指标
            train_loss, train_f1, train_acc, train_precision, train_recall, train_auc = calculate_metrics(
                model, features, edge_index, edge_type,
                train_cases, train_labels, BATCH_SIZE, device
            )
            # 计算验证集指标
            val_loss, val_f1, val_acc, val_precision, val_recall, val_auc = calculate_metrics(
                model, features, edge_index, edge_type,
                val_cases, val_labels, BATCH_SIZE, device
            )

            # 学习率衰减
            scheduler.step(val_loss)

            # 日志打印
            logger.info(f"Epoch [{epoch + 1}/{MAX_EPOCHS}] | "
                        f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | Train AUC: {train_auc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

            # 早停逻辑（监控验证集F1）
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())  # 保存最优模型
                patience_counter = 0
                logger.info(f"验证集F1提升至 {best_val_f1:.4f}，保存最优模型")
            else:
                patience_counter += 1
                logger.info(f"验证集F1未提升，耐心值：{patience_counter}/{PATIENCE}")

            # 触发早停
            if patience_counter >= PATIENCE:
                logger.info(f"早停触发！最优验证集F1：{best_val_f1:.4f}，停止训练")
                break

    # 加载最优模型权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    logger.info("=" * 80)
    logger.info("模型训练完成（加载最优验证集模型）")
    logger.info("=" * 80)
    return model


# ===================== 8. 评估函数 =====================
def evaluate_model(test_cases, labels, model, features, edge_index, edge_type, batch_size, device, threshold):
    model.eval()
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1 = 0.0
    acc = 0.0
    precision = 0.0
    recall = 0.0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    gnn_list = []

    with torch.no_grad():
        for iteration in range(test_batch_num):
            i_start = iteration * batch_size
            i_end = min((iteration + 1) * batch_size, len(test_cases))
            batch_nodes = test_cases[i_start:i_end]
            batch_label = labels[i_start:i_end]

            index = torch.LongTensor(batch_nodes).to(device)
            batch_label_tensor = torch.LongTensor(batch_label).to(device)

            batch_edge_index, _ = get_subgraph_batch(index, edge_index, edge_type)
            batch_edge_index = batch_edge_index.to(device)

            out = model(features.index_select(0, index), batch_edge_index)
            pos_score = F.softmax(out, dim=1)
            prob = pos_score.data.cpu().numpy().argmax(axis=1)
            pos_prob = pos_score.data.cpu().numpy()[:, 1]

            f1 += f1_score(batch_label, prob, average="macro", zero_division=1)
            acc += accuracy_score(batch_label, prob)
            precision += precision_score(batch_label, prob, average="macro", zero_division=1)
            recall += recall_score(batch_label, prob, average="macro", zero_division=1)

            tn, fp, fn, tp = confusion_matrix(batch_label, prob, labels=[0, 1]).ravel()
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_tp += tp

            gnn_list.extend(pos_prob.tolist())

    avg_f1 = f1 / test_batch_num
    avg_acc = acc / test_batch_num
    avg_precision = precision / test_batch_num
    avg_recall = recall / test_batch_num
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
    auc = roc_auc_score(labels, np.array(gnn_list))

    logger.info("=" * 80)
    logger.info("YelpChi数据集模型评估结果")
    logger.info(f"Macro F1: {avg_f1:.4f}")
    logger.info(f"Accuracy: {avg_acc:.4f}")
    logger.info(f"Precision: {avg_precision:.4f}")
    logger.info(f"Recall: {avg_recall:.4f}")
    logger.info(f"FPR: {fpr:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    logger.info("=" * 80)

    return {
        "f1": avg_f1, "acc": avg_acc, "precision": avg_precision,
        "recall": avg_recall, "fpr": fpr, "auc": auc
    }


# ===================== 9. 主函数 =====================
if __name__ == "__main__":
    # 清空旧日志文件
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("")

    # 加载数据集（含验证集）
    features, edge_index, edge_type, train_cases, train_labels, val_cases, val_labels, test_cases, test_labels = load_aml_dataset()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 特征/边移至设备
    features = features.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # 模型初始化（抗过拟合版本）
    model = SelfAttentionGNN(
        in_channels=features.shape[1],
        hidden_sizes=HIDDEN_SIZES,
        out_channels=2,
        dropout=DROPOUT_RATE
    ).to(device)

    # 训练模型（含早停）
    model = train_model(
        model, features, edge_index, edge_type,
        train_cases, train_labels, val_cases, val_labels, device
    )

    # 评估模型
    metrics = evaluate_model(
        test_cases, test_labels, model, features,
        edge_index, edge_type, BATCH_SIZE, device, THRESHOLD
    )

    # 保存最优模型
    torch.save(model.state_dict(), "self_attention_gnn_yelpchi_anti_overfit.pth")
    logger.info(f"最优模型已保存至: self_attention_gnn_yelpchi_anti_overfit.pth")
    logger.info(f"所有日志已保存至: {LOG_FILE_PATH}")