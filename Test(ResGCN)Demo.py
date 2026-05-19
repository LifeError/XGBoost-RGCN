import time
import os
import random
import numpy as np
import scipy.sparse as sp
import copy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 全局配置
DEBUG = False
VERBOSE = True
seed = 42
batch_size = 512
lr = 0.001
min_lr = 1e-7  # 提高学习率下限
num_epochs = 5000  # 增加最大训练轮数
test_epochs = 20  # 增加测试频率
hidden_sizes = [512, 512, 512]  # 增加隐藏层大小
no_cuda = 1
patience = 30  # 增加早停容忍度
l2_reg = 0.0001
accumulation_steps = 4  # 梯度累积步数

# 设置随机种子
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# 数据处理函数
def normalize(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def pos_neg_split(nodes, labels):
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(nodes[idx])
            neg_nodes.remove(nodes[idx])
    return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
    aux_nodes = random.sample(neg_nodes, k=int(len(pos_nodes) * scale))
    return pos_nodes + aux_nodes


# 残差GCN模型
class ResGCN(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features):
        super(ResGCN, self).__init__()

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 残差块
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            block = nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_sizes[i + 1], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.res_blocks.append(block)

        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], out_features)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # 残差连接

        x = self.output_layer(x)
        return x


# 评估函数
def test_model(test_cases, labels, model, features, batch_size):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        all_scores = []

        for i in range(0, len(test_cases), batch_size):
            batch_nodes = test_cases[i:i + batch_size]
            batch_label = labels[i:i + batch_size]
            index = torch.LongTensor(batch_nodes).cuda() if cuda else torch.LongTensor(batch_nodes)

            out = model(features(index))
            scores = torch.softmax(out, dim=1)
            preds = scores.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch_label)
            all_scores.extend(scores[:, 1].cpu().numpy())

        # 计算指标
        f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average="macro")

        # 计算FPR
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        auc = roc_auc_score(all_labels, all_scores)

        return {
            "f1": f1,
            "acc": acc,
            "recall": recall,
            "fpr": fpr,
            "auc": auc
        }


# 加载数据
prefix = 'work/'
data_file = loadmat(prefix + 'YelpChi.mat')
labels = data_file['label'].flatten()
feat_data = data_file['features'].todense().A

# 划分数据集
index = list(range(len(labels)))
idx_train, idx_test, y_train, y_test = train_test_split(
    index, labels, stratify=labels, test_size=0.6, random_state=2
)
train_pos, train_neg = pos_neg_split(idx_train, y_train)

# 设置设备和特征
cuda = not no_cuda and torch.cuda.is_available()
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
features.weight = nn.Parameter(torch.FloatTensor(normalize(feat_data)), requires_grad=False)
if cuda:
    features.cuda()

# 初始化模型和优化器
model = ResGCN(in_features=feat_data.shape[1], hidden_sizes=hidden_sizes, out_features=2)
if cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
criterion = nn.CrossEntropyLoss()

# 学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

# 训练循环
best_f1 = 0
best_loss = float('inf')
no_improve_count = 0

for epoch in range(num_epochs):
    model.train()
    sampled_idx_train = undersample(train_pos, train_neg, scale=1)
    random.shuffle(sampled_idx_train)
    num_batches = len(sampled_idx_train) // batch_size + 1

    total_loss = 0
    start_time = time.time()

    optimizer.zero_grad()  # 梯度清零

    for batch in range(num_batches):
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, len(sampled_idx_train))
        batch_nodes = sampled_idx_train[i_start:i_end]
        batch_label = labels[batch_nodes]

        index = torch.LongTensor(batch_nodes).cuda() if cuda else torch.LongTensor(batch_nodes)
        label = torch.cuda.LongTensor(batch_label) if cuda else torch.LongTensor(batch_label)

        out = model(features(index))
        loss = criterion(out, label)
        loss = loss / accumulation_steps  # 梯度累积：损失平均
        loss.backward()

        # 梯度累积：每accumulation_steps步更新一次参数
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # 还原损失值

    end_time = time.time()

    # 更新学习率
    scheduler.step()

    if VERBOSE and epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch: {epoch}, Loss: {total_loss / num_batches:.6f}, Time: {end_time - start_time:.2f}s, LR: {current_lr:.8f}")

    # 每test_epochs轮评估一次
    if epoch % test_epochs == 0:
        metrics = test_model(idx_test, y_test, model, features, batch_size)
        current_f1 = metrics["f1"]
        current_loss = total_loss / num_batches

        print(f"Epoch {epoch} metrics:")
        print(f"Macro F1: {current_f1:.4f}")
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"FPR: {metrics['fpr']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")

        # 早停机制（同时考虑F1和损失）
        f1_improved = current_f1 > best_f1
        loss_improved = current_loss < best_loss

        if f1_improved or loss_improved:
            best_f1 = max(current_f1, best_f1)
            best_loss = min(current_loss, best_loss)
            no_improve_count = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved. Best F1: {best_f1:.4f}, Best Loss: {best_loss:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch}. Best F1: {best_f1:.4f}")
                break

# 加载最佳模型并进行最终评估
print("Loading best model...")
model.load_state_dict(torch.load('best_model.pth'))
print("Final evaluation on test set:")
metrics = test_model(idx_test, y_test, model, features, batch_size)
print(f"Macro F1: {metrics['f1']:.4f}")
print(f"Accuracy: {metrics['acc']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"FPR: {metrics['fpr']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")