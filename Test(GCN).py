import time
import os
import random
import pickle
import numpy as np
import scipy.sparse as sp
import copy as cp

import sklearn.metrics
import torch
import torch.nn as nn
from torch.nn import init
from scipy.io import loadmat
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import logging
import sys


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
            logging.FileHandler('Test(GCN)_logging_results.txt')  # 文件输出
        ]
    )


# 立即设置日志配置
setup_logging()

# 测试日志
logging.info("日志系统已成功配置")

# 全局初始化配置参数，固定随机种子
DEBUG = False  # Debug模式可快速跑通代码，非Debug模式可得到更好的结果
VERBOSE = True  # Verbose模式打印更多参数
seed = 42
batch_size = 512
lr = 0.1
num_epochs = 2 if DEBUG else 100
test_epochs = 10
emb_size = 32
hidden_size = 128
no_cuda = 1


# 矩阵正则化
def normalize(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# 将数据集按照类别的正负分开
def pos_neg_split(nodes, labels):
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


# 设计评估指标和评估函数
def test_model(test_cases, labels, model, batch_size):
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1 = 0.0
    acc = 0.0
    gnn_list = []
    recall = 0.0
    device = next(model.parameters()).device  # 获取模型所在设备

    # 对每一个测试batch进行遍历
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]

        # 创建 index 张量并移至模型所在设备
        if isinstance(batch_nodes, list):
            index = torch.LongTensor(batch_nodes).to(device)
        else:
            index = batch_nodes.to(device)

        out = model(features(index))

        # 每个测试节点被分类为正类的概率
        pos_score = torch.sigmoid(out)
        prob = pos_score.data.cpu().numpy().argmax(axis=1)
        recall = recall_score(batch_label,prob)
        # 计算F1-macro分数
        f1 += f1_score(batch_label, prob, average="macro")
        # 计算准确度分数
        acc += accuracy_score(batch_label, prob)
        # 记录每个节点被分为正类的概率,用来计算AUC分数
        gnn_list.extend(pos_score.data.cpu().numpy()[:, 1].tolist())

    # 计算AUC分数
    auc = roc_auc_score(labels, np.array(gnn_list))
    logging.info(f"Macro F1: {f1 / test_batch_num:.4f}")
    logging.info(f"Accuracy: {acc / test_batch_num:.4f}")
    logging.info(f"AUC: {auc:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"AUC: {auc:.4f}")


# 对负样本随机降采样,scale是采样后的负类与正类的比例,这里采用1:1
def undersample(pos_nodes, neg_nodes, scale=1):
    aux_nodes = cp.deepcopy(neg_nodes)
    aux_nodes = random.sample(aux_nodes, k=int(len(pos_nodes) * scale))
    # 得到训练batch的节点集合
    batch_nodes = pos_nodes + aux_nodes

    return batch_nodes


# 数据加载
prefix = 'work/'
# 读入mat格式的数据
data_file = loadmat(prefix + 'YelpChi.mat')
labels = data_file['label'].flatten()
feat_data = data_file['features'].todense().A
# 计算数据的类别不平衡程度
logging.info('Class imbalance ratio: {}'.format(np.mean(labels)))

# 设置随机数种子
np.random.seed(seed)
random.seed(seed)
index = list(range(len(labels)))
# 划分训练集和测试集
idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.60, random_state=2,
                                                        shuffle=True)
train_pos, train_neg = pos_neg_split(idx_train, y_train)

# 查看是有可用的cuda
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# 创建特征嵌入并移至指定设备
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
features = features.to(device)

# 构建一个含有线性层和激活层的简单MLP
net = torch.nn.Sequential(
    nn.Linear(emb_size, hidden_size),
    nn.Tanh(),
    nn.Linear(hidden_size, 2)
)

net = net.to(device)

# 设置损失函数为交叉熵损失
loss_function = nn.CrossEntropyLoss()
# 设置优化器为SGD
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# 训练模型
for epoch in range(num_epochs):
    # 在每个epoch中,随机降采样负类样本
    sampled_idx_train = undersample(train_pos, train_neg, scale=1)
    random.shuffle(sampled_idx_train)
    num_batches = int(len(sampled_idx_train) / batch_size) + 1

    loss = 0.0
    epoch_time = 0

    # mini-batch训练
    for batch in range(num_batches):
        start_time = time.time()
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, len(sampled_idx_train))
        # 获取降采样后的节点和其标签
        batch_nodes = sampled_idx_train[i_start:i_end]
        batch_label = labels[np.array(batch_nodes)]

        # 创建 index 张量并移至模型所在设备
        if isinstance(batch_nodes, list):
            index = torch.LongTensor(batch_nodes).to(device)
        else:
            index = batch_nodes.to(device)

        # 计算模型输出
        out = net(features(index))

        # 创建 label 张量并移至模型所在设备
        label = torch.LongTensor(batch_label).to(device)

        # 计算分类损失
        loss = loss_function(out, label)

        # 用优化器进行梯度回传.更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录当前batch的训练时间和分类损失
        end_time = time.time()
        epoch_time += end_time - start_time
        loss += loss.item()

    # 输出当前batch的详细情况
    if VERBOSE:
        logging.info(f'Epoch: {epoch}, loss: {loss / num_batches:.4f}, time: {epoch_time:.2f}s')

    # 在测试集上测试模型的表现
    if epoch % test_epochs == 0:
        test_model(idx_test, y_test, net, batch_size)