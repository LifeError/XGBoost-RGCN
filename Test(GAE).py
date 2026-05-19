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
from torch import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.utils import to_undirected, subgraph, negative_sampling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GAE
import logging

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Test(GAE)_logging_results.txt'),
        logging.StreamHandler()
    ]
)

# Configuration
seed = 42
batch_size = 512
lr = 0.001
min_lr = 1e-7
num_epochs = 5000
test_epochs = 20
hidden_sizes = [256, 256, 256]
no_cuda = 1
patience = 10
l2_reg = 0.0001
accumulation_steps = 4

# Reproducibility
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Normalization function
def normalize(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


# Split positive/negative
def pos_neg_split(nodes, labels):
    pos_nodes, neg_nodes = [], []
    for idx in nodes:
        if labels[idx] == 1:
            pos_nodes.append(idx)
        else:
            neg_nodes.append(idx)
    return pos_nodes, neg_nodes


# Undersample
def undersample(pos_nodes, neg_nodes, scale=1):
    sampled_neg = random.sample(neg_nodes, k=int(len(pos_nodes) * scale))
    return pos_nodes + sampled_neg


# Encoder for GAE
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_sizes[0])
        self.convs = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.convs.append(GCNConv(hidden_sizes[i], hidden_sizes[i + 1]))
        self.conv_out = GCNConv(hidden_sizes[-1], out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.conv_out(x, edge_index)


# Batch-wise evaluation with proper subgraph handling
def evaluate_model(model, features, edge_index, labels, test_nodes, batch_size, cuda=False):
    model.eval()
    all_preds, all_targets, all_scores = [], [], []

    # 确保labels是NumPy数组
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # 分批处理测试节点
    for i in range(0, len(test_nodes), batch_size):
        batch_nodes = test_nodes[i:i + batch_size]

        # 创建当前批次的子图
        batch_edge_index, _ = subgraph(
            batch_nodes,
            edge_index,
            relabel_nodes=True,
            num_nodes=features.size(0)
        )

        if cuda:
            batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long).cuda()
            batch_edge_index = batch_edge_index.cuda()
        else:
            batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long)

        with torch.no_grad():
            # 获取批次节点的特征
            batch_features = features[batch_nodes_tensor]

            # 编码节点
            batch_emb = model.encode(batch_features, batch_edge_index)

            # 使用节点嵌入进行分类
            scores = torch.norm(batch_emb, dim=1)
            preds = (scores > scores.mean()).float().cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels[batch_nodes])  # 使用原始batch_nodes索引labels
            all_scores.extend(scores.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average="macro")
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=1)
    recall = recall_score(all_targets, all_preds, average="macro")
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(all_targets, all_scores)

    return {"f1": f1, "precision": precision, "recall": recall, "fpr": fpr, "auc": auc}


# Load data
prefix = 'work/'
data = loadmat(prefix + 'YelpChi.mat')

# 查看数据结构
logging.info("数据文件包含的字段: %s", list(data.keys()))
logging.info("homo shape: %s %s", data['homo'].shape, type(data['homo']))
logging.info("net_rur shape: %s %s", data['net_rur'].shape, type(data['net_rur']))
logging.info("net_rtr shape: %s %s", data['net_rtr'].shape, type(data['net_rtr']))
logging.info("net_rsr shape: %s %s", data['net_rsr'].shape, type(data['net_rsr']))

labels = data['label'].flatten()
feat_data = data['features'].todense().A
feat_data = normalize(feat_data)

# 选择使用的邻接矩阵
adj_matrix = data['homo']  # 使用homo作为默认邻接矩阵

# 构建边索引
logging.info("正在构建边索引...")
if sp.issparse(adj_matrix):
    adj = adj_matrix.tocoo()
    edge_index = torch.tensor(np.stack([adj.row, adj.col]), dtype=torch.long)
else:
    edge_index = torch.tensor(np.nonzero(adj_matrix), dtype=torch.long)

# 确保图是无向的
edge_index = to_undirected(edge_index)
logging.info(f"边索引构建完成，形状: {edge_index.shape}")

# Split dataset
index = list(range(len(labels)))
idx_train, idx_test = train_test_split(index, stratify=labels, test_size=0.6, random_state=2)
train_pos, train_neg = pos_neg_split(idx_train, labels)

# Setup CUDA and features
cuda = not no_cuda and torch.cuda.is_available()
features = torch.FloatTensor(feat_data)
if cuda:
    features = features.cuda()

# 创建训练节点的数据加载器
train_nodes = torch.tensor(undersample(train_pos, train_neg, scale=1), dtype=torch.long)
train_loader = DataLoader(train_nodes, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
encoder = GCNEncoder(in_channels=feat_data.shape[1], out_channels=hidden_sizes[-1])
model = GAE(encoder)
if cuda:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

# Early stopping setup
best_f1 = 0
best_loss = float('inf')
no_improve = 0
best_model_state = None

# Mixed precision setup
if cuda:
    scaler = GradScaler()

# Training loop with mini-batches
logging.info("开始训练...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batches = 0

    for batch_nodes in train_loader:
        if cuda:
            batch_nodes = batch_nodes.cuda()

        # 为当前批次节点创建子图
        batch_edge_index, _ = subgraph(
            batch_nodes,
            edge_index,
            relabel_nodes=True,
            num_nodes=features.size(0)
        )

        # 只将当前批次的特征移到GPU
        batch_features = features[batch_nodes]

        optimizer.zero_grad()

        if cuda:
            with autocast():
                z = model.encode(batch_features, batch_edge_index)
                # 采样负边
                neg_edge_index = negative_sampling(
                    batch_edge_index,
                    num_nodes=batch_nodes.size(0),
                    num_neg_samples=batch_edge_index.size(1)
                )
                loss = model.recon_loss(z, batch_edge_index, neg_edge_index) / accumulation_steps
        else:
            z = model.encode(batch_features, batch_edge_index)
            neg_edge_index = negative_sampling(
                batch_edge_index,
                num_nodes=batch_nodes.size(0),
                num_neg_samples=batch_edge_index.size(1)
            )
            loss = model.recon_loss(z, batch_edge_index, neg_edge_index) / accumulation_steps

        if cuda:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batches + 1) % accumulation_steps == 0:
            if cuda:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
        batches += 1

        # 释放不再需要的张量
        del batch_features, batch_edge_index, z, loss, neg_edge_index
        if cuda:
            torch.cuda.empty_cache()

    scheduler.step()

    if epoch % test_epochs == 0:
        # 使用优化后的评估函数
        metrics = evaluate_model(model, features, edge_index, labels, idx_test, batch_size, cuda)

        logging.info(
            f"Epoch {epoch} | Loss: {epoch_loss / batches:.4f} | F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | FPR: {metrics['fpr']:.4f} | AUC: {metrics['auc']:.4f}")

        # Early stopping check
        if metrics['f1'] > best_f1 or epoch_loss / batches < best_loss:
            best_f1 = max(best_f1, metrics['f1'])
            best_loss = min(best_loss, epoch_loss / batches)
            no_improve = 0
            best_model_state = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

# After training, load the best model and test
logging.info("Training complete, loading best model...")
model.load_state_dict(best_model_state)

# Final evaluation
final_metrics = evaluate_model(model, features, edge_index, labels, idx_test, batch_size, cuda)
logging.info("Final Evaluation after training:")
logging.info(f"F1: {final_metrics['f1']:.4f}")
logging.info(f"Precision: {final_metrics['precision']:.4f}")
logging.info(f"Recall: {final_metrics['recall']:.4f}")
logging.info(f"FPR: {final_metrics['fpr']:.4f}")
logging.info(f"AUC: {final_metrics['auc']:.4f}")