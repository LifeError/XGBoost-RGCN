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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import logging

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
            logging.FileHandler('Test(ResGCN)_logging_results.txt')  # 文件输出
        ]
    )


# 立即设置日志配置
setup_logging()

# 测试日志
logging.info("日志系统已成功配置")

# Configuration
seed = 42
batch_size = 512
lr = 0.001
min_lr = 1e-7
num_epochs = 5000
test_epochs = 20
hidden_sizes = [512, 512, 512]
no_cuda = 1
patience = 10  # Early stopping patience
l2_reg = 0.0001
accumulation_steps = 4

# Reproducibility
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 记录程序启动信息
logging.info("Program started.")

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

# ResGCN model with LeakyReLU
class ResGCN(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features):
        super(ResGCN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2)
        )
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            block = nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.2),
                nn.Linear(hidden_sizes[i + 1], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.2)
            )
            self.res_blocks.append(block)
        self.output_layer = nn.Linear(hidden_sizes[-1], out_features)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Skip connection (residual connection)
        return self.output_layer(x)

# Evaluation
def evaluate_model(test_nodes, all_labels, model, features, batch_size, cuda=False):
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
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_label)
            all_scores.extend(probs[:, 1].cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average="macro")
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=1)
    recall = recall_score(all_targets, all_preds, average="macro")
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(all_targets, all_scores)

    return {"f1": f1, "precision": precision, "recall": recall, "fpr": fpr, "auc": auc}

# Load data
prefix = 'work/'
try:
    logging.info("Starting to load data.")
    data = loadmat(prefix + 'YelpChi.mat')
    labels = data['label'].flatten()
    feat_data = data['features'].todense().A
    feat_data = normalize(feat_data)
    logging.info("Data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Split dataset
index = list(range(len(labels)))
idx_train, idx_test = train_test_split(index, stratify=labels, test_size=0.6, random_state=2)
train_pos, train_neg = pos_neg_split(idx_train, labels)

# Setup CUDA and features
cuda = not no_cuda and torch.cuda.is_available()
features = nn.Embedding.from_pretrained(torch.FloatTensor(feat_data), freeze=True)
if cuda:
    features = features.cuda()
    logging.info("CUDA is available and features moved to GPU.")
else:
    logging.info("CUDA is not available, using CPU.")

# Initialize model and optimizer
model = ResGCN(in_features=feat_data.shape[1], hidden_sizes=hidden_sizes, out_features=2)
if cuda:
    model = model.cuda()
    logging.info("Model moved to GPU.")
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
criterion = nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

# Early stopping setup
best_f1 = 0
best_loss = float('inf')
no_improve = 0
best_model_state = None

# Mixed precision setup
if cuda:  # Only use mixed precision if CUDA is available
    scaler = GradScaler()
    logging.info("Mixed precision training enabled.")

# Training loop with early stopping
logging.info("Starting training loop.")
for epoch in range(num_epochs):
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

        # Only use autocast if CUDA is available
        if cuda:
            with autocast():  # Mixed precision
                out = model(features(index))
                loss = criterion(out, label) / accumulation_steps
        else:
            out = model(features(index))
            loss = criterion(out, label) / accumulation_steps

        if cuda:
            scaler.scale(loss).backward()  # Backprop with scaled loss
        else:
            loss.backward()  # No scaling for CPU

        if (i // batch_size + 1) % accumulation_steps == 0:
            if cuda:
                scaler.step(optimizer)  # Optimizer step with mixed precision
                scaler.update()  # Update the scaler
            else:
                optimizer.step()  # Standard optimizer step for CPU
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    scheduler.step()
    if epoch % test_epochs == 0:
        metrics = evaluate_model(idx_test, labels, model, features, batch_size, cuda)
        logging.info(
            f"Epoch {epoch} | Loss: {total_loss:.4f} | F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | FPR: {metrics['fpr']:.4f} | AUC: {metrics['auc']:.4f}")

        # Early stopping check
        if metrics['f1'] > best_f1 or total_loss < best_loss:
            best_f1 = max(best_f1, metrics['f1'])
            best_loss = min(best_loss, total_loss)
            no_improve = 0
            best_model_state = model.state_dict()  # Save the best model state
            logging.info(f"Best model state updated at epoch {epoch}.")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

# After training, load the best model and test
logging.info("Training complete, loading best model...")
model.load_state_dict(best_model_state)

# Final evaluation on the test set after training is complete
final_metrics = evaluate_model(idx_test, labels, model, features, batch_size, cuda)
logging.info("Final Evaluation after training:")
logging.info(f"F1: {final_metrics['f1']:.4f}")
logging.info(f"Precision: {final_metrics['precision']:.4f}")
logging.info(f"Recall: {final_metrics['recall']:.4f}")
logging.info(f"FPR: {final_metrics['fpr']:.4f}")
logging.info(f"AUC: {final_metrics['auc']:.4f}")

print("Final Evaluation after training:")
print(f"F1: {final_metrics['f1']:.4f}")
print(f"Precision: {final_metrics['precision']:.4f}")
print(f"Recall: {final_metrics['recall']:.4f}")
print(f"FPR: {final_metrics['fpr']:.4f}")
print(f"AUC: {final_metrics['auc']:.4f}")