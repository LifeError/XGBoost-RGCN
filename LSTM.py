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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# 配置参数
seed = 42
batch_size = 512
lr = 0.001
min_lr = 1e-7
num_epochs = 5000
test_epochs = 20
hidden_sizes = [512, 512, 512]
no_cuda = 1
patience = 15
l2_reg = 0.0001
accumulation_steps = 4
early_stop_metric = 'f1'  # 可选 'f1', 'loss', 'both'

# 设置随机种子以保证结果可复现
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# 对矩阵进行归一化处理
def normalize(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


# 将节点根据标签分为正样本节点和负样本节点
def pos_neg_split(nodes, labels):
    pos_nodes, neg_nodes = [], []
    for idx in nodes:
        if labels[idx] == 1:
            pos_nodes.append(idx)
        else:
            neg_nodes.append(idx)
    return pos_nodes, neg_nodes


# 对负样本进行欠采样
def undersample(pos_nodes, neg_nodes, scale=1):
    sampled_neg = random.sample(neg_nodes, k=int(len(pos_nodes) * scale))
    return pos_nodes + sampled_neg


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_sizes[0], num_layers=1, batch_first=True)
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fc_layers.append(nn.LeakyReLU(negative_slope=0.2))
            self.fc_layers.append(nn.Dropout(0.2))
        self.output_layer = nn.Linear(hidden_sizes[-1], out_features)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加序列维度
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        for layer in self.fc_layers:
            x = layer(x)
        return self.output_layer(x)


# 评估模型性能
def evaluate_model(test_nodes, all_labels, model, features, batch_size, cuda=False, threshold=0.5):
    model.eval()
    all_preds, all_targets, all_scores = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            batch_nodes = test_nodes[i:i + batch_size]
            index = torch.LongTensor(batch_nodes).cuda() if cuda else torch.LongTensor(batch_nodes)
            batch_label = all_labels[batch_nodes]
            label = torch.LongTensor(batch_label).cuda() if cuda else torch.LongTensor(batch_label)
            out = model(features(index))
            probs = torch.softmax(out, dim=1)
            preds = (probs[:, 1] > threshold).cpu().numpy().astype(int)  # 使用阈值
            all_preds.extend(preds)
            all_targets.extend(batch_label)
            all_scores.extend(probs[:, 1].cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average="macro")
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=1)
    recall = recall_score(all_targets, all_preds, average="macro")
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(all_targets, all_scores)

    return {"f1": f1, "precision": precision, "recall": recall, "fpr": fpr, "auc": auc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# 加载数据
prefix = 'work/'
data = loadmat(prefix + 'YelpChi.mat')
labels = data['label'].flatten()
feat_data = data['features'].todense().A
feat_data = normalize(feat_data)

# 划分数据集
index = list(range(len(labels)))
idx_train, idx_test = train_test_split(index, stratify=labels, test_size=0.6, random_state=2)
train_pos, train_neg = pos_neg_split(idx_train, labels)

# 设置CUDA和特征嵌入
cuda = not no_cuda and torch.cuda.is_available()
features = nn.Embedding.from_pretrained(torch.FloatTensor(feat_data), freeze=True)
if cuda:
    features = features.cuda()

# 初始化模型和优化器
model = LSTMModel(in_features=feat_data.shape[1], hidden_sizes=hidden_sizes, out_features=2)
if cuda:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
criterion = nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

# 早停策略设置
best_f1 = 0
best_loss = float('inf')
best_auc = 0
no_improve = 0
best_model_state = None
best_epoch = 0

# 混合精度训练设置
if cuda:  # 仅在CUDA可用时使用混合精度训练
    scaler = GradScaler()

# 打开文件以保存输出结果
with open('LSTM_results.txt', 'w') as output_file:
    # 写入配置信息
    config_str = (f"Configuration:\n"
                  f"Seed={seed}, Batch Size={batch_size}, LR={lr}, Min LR={min_lr}, "
                  f"Epochs={num_epochs}, Hidden Sizes={hidden_sizes}, "
                  f"CUDA={cuda}, Patience={patience}, L2 Reg={l2_reg}, "
                  f"Accumulation Steps={accumulation_steps}, Early Stop Metric={early_stop_metric}\n")
    print(config_str)
    output_file.write(config_str + '\n')

    # 训练循环，包含早停策略
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        sampled_train = undersample(train_pos, train_neg, scale=1)
        random.shuffle(sampled_train)
        total_loss = 0
        optimizer.zero_grad()

        for i in range(0, len(sampled_train), batch_size):
            batch_nodes = sampled_train[i:i + batch_size]
            batch_label = labels[batch_nodes]
            index = torch.LongTensor(batch_nodes).cuda() if cuda else torch.LongTensor(batch_nodes)
            label = torch.LongTensor(batch_label).cuda() if cuda else torch.LongTensor(batch_label)

            # 仅在CUDA可用时使用自动混合精度
            if cuda:
                with autocast():  # 混合精度训练
                    out = model(features(index))
                    loss = criterion(out, label) / accumulation_steps
            else:
                out = model(features(index))
                loss = criterion(out, label) / accumulation_steps

            if cuda:
                scaler.scale(loss).backward()  # 反向传播，使用缩放后的损失
            else:
                loss.backward()  # CPU不使用缩放

            if (i // batch_size + 1) % accumulation_steps == 0:
                if cuda:
                    scaler.step(optimizer)  # 优化器更新，使用混合精度
                    scaler.update()  # 更新缩放器
                else:
                    optimizer.step()  # CPU使用标准优化器更新
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 每test_epochs轮评估一次
        if epoch % test_epochs == 0:
            metrics = evaluate_model(idx_test, labels, model, features, batch_size, cuda)

            # 输出详细日志
            log_str = (f"Epoch {epoch} | Time: {time.time() - start_time:.2f}s | "
                       f"Loss: {total_loss:.4f} | LR: {current_lr:.8f} | "
                       f"F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | "
                       f"Recall: {metrics['recall']:.4f} | FPR: {metrics['fpr']:.4f} | "
                       f"AUC: {metrics['auc']:.4f} | "
                       f"TP: {metrics['tp']} | FP: {metrics['fp']} | TN: {metrics['tn']} | FN: {metrics['fn']} | "
                       f"No Improve: {no_improve}/{patience} | "
                       f"Best F1: {best_f1:.4f} | Best Loss: {best_loss:.4f} | Best AUC: {best_auc:.4f}")
            print(log_str)
            output_file.write(log_str + '\n')

            # 早停检查 - 根据配置的指标决定
            metric_improved = False

            if early_stop_metric == 'f1':
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    metric_improved = True
            elif early_stop_metric == 'loss':
                if total_loss < best_loss:
                    best_loss = total_loss
                    metric_improved = True
            elif early_stop_metric == 'both':
                f1_improved = metrics['f1'] > best_f1
                loss_improved = total_loss < best_loss
                if f1_improved and loss_improved:
                    best_f1 = metrics['f1']
                    best_loss = total_loss
                    metric_improved = True
            elif early_stop_metric == 'auc':
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    metric_improved = True

            if metric_improved:
                no_improve = 0
                best_model_state = model.state_dict()
                best_epoch = epoch
                log_str = f"Epoch {epoch}: New best model saved! Best {early_stop_metric}: {best_f1 if early_stop_metric == 'f1' else best_loss if early_stop_metric == 'loss' else best_auc:.4f}"
                print(log_str)
                output_file.write(log_str + '\n')
            else:
                no_improve += 1
                log_str = f"Epoch {epoch}: No improvement ({no_improve}/{patience})"
                print(log_str)
                output_file.write(log_str + '\n')
                if no_improve >= patience:
                    log_str = f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}."
                    print(log_str)
                    output_file.write(log_str + '\n')
                    break

    # 训练结束后，加载最佳模型并进行测试
    log_str = "Training complete, loading best model..."
    print(log_str)
    output_file.write(log_str + '\n')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        log_str = "Warning: No best model found! Using the last model state."
        print(log_str)
        output_file.write(log_str + '\n')

    # 最终测试集评估
    log_str = "\nFinal Evaluation after training:"
    print(log_str)
    output_file.write(log_str + '\n')

    # 在不同阈值下评估模型（适用于反洗钱场景）
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        final_metrics = evaluate_model(idx_test, labels, model, features, batch_size, cuda, threshold)
        log_str = (f"\nThreshold = {threshold}:\n"
                   f"F1: {final_metrics['f1']:.4f} | "
                   f"Precision: {final_metrics['precision']:.4f} | "
                   f"Recall: {final_metrics['recall']:.4f} | "
                   f"FPR: {final_metrics['fpr']:.4f} | "
                   f"AUC: {final_metrics['auc']:.4f} | "
                   f"TP: {final_metrics['tp']} | FP: {final_metrics['fp']} | "
                   f"TN: {final_metrics['tn']} | FN: {final_metrics['fn']}")
        print(log_str)
        output_file.write(log_str + '\n')

    # 推荐最佳阈值（基于F1分数）
    best_threshold = max(thresholds,
                         key=lambda t: evaluate_model(idx_test, labels, model, features, batch_size, cuda, t)['f1'])
    log_str = f"\nRecommended threshold for highest F1: {best_threshold}"
    print(log_str)
    output_file.write(log_str + '\n')