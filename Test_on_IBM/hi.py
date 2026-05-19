import sys
import time
import random
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import xgboost as xgb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import subgraph
import logging
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ===================== 轻量化全局参数（重点修改） =====================
seed = 42
batch_size = 256  # 从1024降到256，大幅降低显存占用
lr = 0.001
min_lr = 1e-7
num_epochs = 1000  # 从5000降到1000，减少训练时间
test_epochs = 20
hidden_sizes = [64, 64, 64]  # 从128降到64，减少模型参数
no_cuda = False
patience = 10
l2_reg = 0.0001
accumulation_steps = 2  # 梯度累积步数减半
dropout_rate = 0.3  # 降低dropout，避免小数据量下欠拟合
recall_weight = 2.0
fpr_weight = 1.0
threshold = 0.7
thresholds_to_test = [0.3, 0.5, 0.7, 0.9]  # 减少测试阈值数量
sample_ratio = 0.3  # 节点采样比例：保留30%的账户（可根据GPU调整，显存<4G设0.2，4-8G设0.5）
edge_sample_num = 20  # 每个节点最多保留20条出边（减少边数量）


# ===================== 日志配置 =====================
def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('RGCN-XGBoost_HI_light_results.txt')
        ]
    )


setup_logging()
logging.info("=" * 60)
logging.info(f"HI数据集轻量化处理（采样比例{sample_ratio}）+ RGCN训练")
logging.info("=" * 60)

# ===================== 随机种子设置 =====================
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== 工具函数 =====================
def calculate_class_weights(labels):
    num_pos = np.sum(labels)
    num_neg = len(labels) - num_pos
    total = num_pos + num_neg
    weight_pos = total / (2 * num_pos) if num_pos > 0 else 1.0
    weight_neg = total / (2 * num_neg) if num_neg > 0 else 1.0
    return torch.tensor([weight_neg, weight_pos], dtype=torch.float)


def print_gpu_memory(device):
    if torch.cuda.is_available():
        logging.info(
            f"GPU内存 - 分配: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB, 缓存: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


def normalize(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


# ===================== 轻量化RGCN模型（隐藏层已缩小） =====================
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


# ===================== 模型评估函数（兼容小数据） =====================
def evaluate_model(test_nodes, all_labels, model, features, edge_index, edge_type, batch_size, device, threshold):
    model.eval()
    all_preds, all_targets, all_scores = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            batch_nodes = test_nodes[i:i + batch_size]
            batch_node_idx = torch.LongTensor(batch_nodes).to(device)

            subgraph_result = subgraph(
                batch_node_idx,
                edge_index,
                edge_attr=edge_type,
                relabel_nodes=True,
                num_nodes=features.size(0)
            )

            if len(subgraph_result) == 3:
                batch_edge_index, _, batch_edge_type = subgraph_result
            else:
                batch_edge_index, edge_mask = subgraph_result
                batch_edge_type = edge_type[edge_mask]

            batch_label = all_labels[batch_nodes]
            label = torch.LongTensor(batch_label).to(device)
            out = model(features[batch_node_idx], batch_edge_index, batch_edge_type)
            probs = torch.softmax(out, dim=1)
            scores = probs[:, 1].cpu().numpy()
            preds = (scores > threshold).astype(int)
            all_preds.extend(preds)
            all_targets.extend(batch_label)
            all_scores.extend(scores)

    if len(set(all_targets)) == 1:
        f1 = precision = recall = 0.0
        tn, fp, fn, tp = (len(all_targets), 0, 0, 0) if all_targets[0] == 0 else (0, 0, len(all_targets), 0)
    else:
        f1 = f1_score(all_targets, all_preds, average="macro")
        precision = precision_score(all_targets, all_preds, average="macro", zero_division=1)
        recall = recall_score(all_targets, all_preds, average="macro")
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(all_targets, all_scores) if len(set(all_targets)) > 1 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "fpr": fpr, "auc": auc, "tpr": tpr}


# ===================== 核心：轻量化数据处理（新增采样逻辑） =====================
def process_hi_dataset_light(accounts_path, trans_path, sample_ratio=0.3, edge_sample_num=20):
    """
    轻量化数据处理，包含节点分层采样+边采样
    :param sample_ratio: 节点保留比例（0-1）
    :param edge_sample_num: 每个节点最多保留的出边数
    """
    # 1. 加载数据
    logging.info("加载并清洗数据...")
    accounts_df = pd.read_csv(accounts_path)
    trans_df = pd.read_csv(trans_path)
    trans_df.columns = ['Timestamp', 'From Bank', 'From Account', 'To Bank', 'To Account',
                        'Amount Received', 'Receiving Currency', 'Amount Paid',
                        'Payment Currency', 'Payment Format', 'Is Laundering']

    # 2. 数据清洗（简化版）
    accounts_df = accounts_df.fillna({'Bank Name': 'Unknown', 'Bank ID': -1, 'Entity Name': 'Unknown', 'Entity ID': -1})
    trans_df = trans_df.fillna({'Amount Received': 0.0, 'Amount Paid': 0.0, 'Is Laundering': 0})
    trans_df['Is Laundering'] = trans_df['Is Laundering'].astype(int)
    trans_df['Amount Received'] = pd.to_numeric(trans_df['Amount Received'], errors='coerce').fillna(0.0)
    trans_df['Amount Paid'] = pd.to_numeric(trans_df['Amount Paid'], errors='coerce').fillna(0.0)

    # 3. 生成账户节点 + 初步标签
    all_accounts = set(accounts_df['Account Number'].unique()) | set(trans_df['From Account'].unique()) | set(
        trans_df['To Account'].unique())
    all_accounts = list(all_accounts)
    launder_from = trans_df[trans_df['Is Laundering'] == 1]['From Account'].unique()
    launder_to = trans_df[trans_df['Is Laundering'] == 1]['To Account'].unique()
    launder_accounts = set(launder_from) | set(launder_to)
    account_labels = [1 if acc in launder_accounts else 0 for acc in all_accounts]

    # 4. 节点分层采样（关键：保证洗钱账户比例）
    logging.info(f"分层采样节点（保留比例{sample_ratio}）...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - sample_ratio, random_state=seed)
    sample_idx, _ = next(sss.split(all_accounts, account_labels))
    sampled_accounts = [all_accounts[i] for i in sample_idx]
    sampled_labels = [account_labels[i] for i in sample_idx]
    # 构建采样后的账户映射
    account2idx = {acc: idx for idx, acc in enumerate(sampled_accounts)}
    n_nodes = len(account2idx)
    logging.info(
        f"采样后节点数: {n_nodes} | 洗钱账户数: {sum(sampled_labels)} | 正常账户数: {n_nodes - sum(sampled_labels)}")

    # 5. 轻量化特征工程（减少特征维度）
    logging.info("构建轻量化节点特征...")
    account_features = defaultdict(lambda: {'Bank ID': -1, 'Entity Type': 'Unknown'})
    for _, row in accounts_df.iterrows():
        acc = row['Account Number']
        if acc not in account2idx:
            continue
        account_features[acc]['Bank ID'] = row['Bank ID']
        entity_name = row['Entity Name']
        if 'Sole Proprietorship' in entity_name:
            account_features[acc]['Entity Type'] = 0
        elif 'Corporation' in entity_name:
            account_features[acc]['Entity Type'] = 1
        elif 'Partnership' in entity_name:
            account_features[acc]['Entity Type'] = 2
        else:
            account_features[acc]['Entity Type'] = 3

    # 交易统计特征（仅保留核心4个）
    trans_stats = defaultdict(lambda: {'total_trans': 0, 'total_out': 0.0, 'total_in': 0.0, 'launder_cnt': 0})
    for _, row in trans_df.iterrows():
        from_acc = row['From Account']
        to_acc = row['To Account']
        if from_acc not in account2idx or to_acc not in account2idx:
            continue
        # 统计核心交易特征
        trans_stats[from_acc]['total_trans'] += 1
        trans_stats[from_acc]['total_out'] += row['Amount Paid']
        trans_stats[from_acc]['launder_cnt'] += row['Is Laundering']
        trans_stats[to_acc]['total_trans'] += 1
        trans_stats[to_acc]['total_in'] += row['Amount Received']

    # 构建特征矩阵（仅8维，大幅降维）
    feature_list = []
    for acc in sampled_accounts:
        feat = [
            account_features[acc]['Bank ID'],
            account_features[acc]['Entity Type'],
            trans_stats[acc]['total_trans'],
            trans_stats[acc]['total_out'],
            trans_stats[acc]['total_in'],
            trans_stats[acc]['launder_cnt'],
            trans_stats[acc]['total_out'] / max(trans_stats[acc]['total_trans'], 1),
            trans_stats[acc]['total_in'] / max(trans_stats[acc]['total_trans'], 1)
        ]
        feature_list.append(feat)
    node_features = np.array(feature_list)
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    logging.info(f"轻量化特征维度: {node_features.shape}")

    # 6. 边采样（每个节点最多保留edge_sample_num条出边）
    logging.info(f"边采样（每个节点最多{edge_sample_num}条出边）...")
    payment_types = trans_df['Payment Format'].unique()
    payment2type = {fmt: idx for idx, fmt in enumerate(payment_types)}
    num_relations = len(payment2type)

    # 先按节点分组边
    node_edges = defaultdict(list)
    for _, row in trans_df.iterrows():
        from_acc = row['From Account']
        to_acc = row['To Account']
        fmt = row['Payment Format']
        if from_acc not in account2idx or to_acc not in account2idx or from_acc == to_acc:
            continue
        node_edges[from_acc].append([account2idx[from_acc], account2idx[to_acc], payment2type.get(fmt, 0)])

    # 对每个节点的边进行采样
    edge_index = []
    edge_type = []
    for acc in node_edges:
        edges = node_edges[acc]
        # 不足则全保留，超过则随机采样
        sample_edges = edges if len(edges) <= edge_sample_num else random.sample(edges, edge_sample_num)
        for e in sample_edges:
            edge_index.append([e[0], e[1]])
            edge_type.append(e[2])

    edge_index = torch.tensor(edge_index).T.long()
    edge_type = torch.tensor(edge_type).long()
    logging.info(f"采样后边数: {edge_index.shape[1]} | 边类型数: {num_relations}")

    return node_features, np.array(sampled_labels), edge_index, edge_type, account2idx


# ===================== 主流程 =====================
if __name__ == "__main__":
    # 1. 数据集路径（替换为实际路径）
    ACCOUNTS_PATH = "HI-Small_accounts.csv"
    TRANS_PATH = "HI-Small_Trans.csv"

    # 2. 轻量化数据处理（核心采样）
    node_features, node_labels, edge_index, edge_type, account2idx = process_hi_dataset_light(
        ACCOUNTS_PATH, TRANS_PATH, sample_ratio=sample_ratio, edge_sample_num=edge_sample_num
    )
    num_relations = len(torch.unique(edge_type)) if edge_type.numel() > 0 else 1

    # 3. 分层划分数据集（小数据更要保证比例）
    logging.info("划分训练/验证/测试集...")
    index = list(range(len(node_labels)))
    idx_train_val, idx_test = train_test_split(
        index, stratify=node_labels, test_size=0.2, random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_train_val, stratify=node_labels[idx_train_val], test_size=0.125, random_state=seed
    )
    logging.info(f"训练集: {len(idx_train)}, 验证集: {len(idx_val)}, 测试集: {len(idx_test)}")

    # 4. XGBoost特征增强（轻量化：减少树数量）
    logging.info("训练轻量化XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=seed, max_depth=4)
    xgb_model.fit(node_features[idx_train], node_labels[idx_train])
    xgb_probs = xgb_model.predict_proba(node_features)
    feat_data_aug = np.hstack([node_features, xgb_probs])
    feat_data = normalize(feat_data_aug)

    # 5. 设备设置（强制CPU如果GPU不足）
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    logging.info(f"使用设备: {device}")
    if device.type == "cpu":
        logging.warning("当前使用CPU训练，速度较慢，建议适当再降低采样比例")

    # 6. 数据转Tensor
    features = torch.FloatTensor(feat_data).to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # 7. 模型初始化
    class_weights = calculate_class_weights(node_labels[idx_train]).to(device)
    model = RGCN(
        in_features=feat_data.shape[1],
        hidden_sizes=hidden_sizes,
        out_features=2,
        num_relations=num_relations
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    # 8. 早停设置
    best_metric = float('-inf')
    no_improve = 0
    best_model_state = None
    scaler = torch.cuda.amp.GradScaler(enabled=True) if device.type == "cuda" else None

    # 9. 轻量化训练循环
    logging.info("开始轻量化训练...")
    for epoch in range(num_epochs):
        model.train()
        random.shuffle(idx_train)
        total_loss = 0
        optimizer.zero_grad()

        for i in range(0, len(idx_train), batch_size):
            batch_nodes = idx_train[i:i + batch_size]
            batch_node_idx = torch.LongTensor(batch_nodes).to(device)

            subgraph_result = subgraph(
                batch_node_idx, edge_index, edge_attr=edge_type, relabel_nodes=True, num_nodes=features.size(0)
            )
            batch_edge_index, batch_edge_type = (subgraph_result[:2] if len(subgraph_result) == 3 else (
            subgraph_result[0], edge_type[subgraph_result[1]]))

            batch_label = node_labels[batch_nodes]
            label = torch.LongTensor(batch_label).to(device)

            # 前向传播（关闭混合精度如果显存极不足）
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = model(features[batch_node_idx], batch_edge_index, batch_edge_type)
                    loss = criterion(out, label) / accumulation_steps
                scaler.scale(loss).backward()
            else:
                out = model(features[batch_node_idx], batch_edge_index, batch_edge_type)
                loss = criterion(out, label) / accumulation_steps
                loss.backward()

            # 梯度累积
            if (i // batch_size + 1) % accumulation_steps == 0:
                if device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        scheduler.step()

        # 验证集评估
        if epoch % test_epochs == 0:
            metrics = evaluate_model(idx_val, node_labels, model, features, edge_index, edge_type, batch_size, device,
                                     threshold)
            current_metric = recall_weight * metrics['recall'] - fpr_weight * metrics['fpr']
            logging.info(
                f"Epoch {epoch} | Loss: {total_loss:.4f} | F1: {metrics['f1']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | FPR: {metrics['fpr']:.4f} | AUC: {metrics['auc']:.4f}"
            )

            if current_metric > best_metric:
                best_metric = current_metric
                no_improve = 0
                best_model_state = model.state_dict()
            else:
                no_improve += 1
                if no_improve >= patience:
                    logging.info(f"早停于Epoch {epoch}")
                    break

    # 10. 最终测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_metrics = evaluate_model(idx_test, node_labels, model, features, edge_index, edge_type, batch_size, device,
                                   threshold)
    logging.info("=" * 60)
    logging.info("最终测试结果:")
    logging.info(
        f"F1: {final_metrics['f1']:.4f} | Recall: {final_metrics['recall']:.4f} | AUC: {final_metrics['auc']:.4f}")

    # 不同阈值测试（简化版）
    logging.info("\n不同阈值测试结果:")
    for thresh in thresholds_to_test:
        metrics = evaluate_model(idx_test, node_labels, model, features, edge_index, edge_type, batch_size, device,
                                 thresh)
        logging.info(f"阈值{thresh}: F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")