import logging

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


seed = 42

def load_elliptic_data():
    """
    加载Elliptic数据集，返回特征、边、标签
    :param data_dir: 数据集存放目录（需提前下载）
    :return: feat_data (np.ndarray), edges (np.ndarray), labels (np.ndarray), node_ids (np.ndarray)
    """
    # 1. 加载节点特征和标签
    nodes_df = pd.read_csv("elliptic_txs_features.csv")
    classes_df = pd.read_csv("elliptic_txs_classes.csv")

    # 列名：0=节点ID，1=时间步，2-167=特征
    node_ids = nodes_df.iloc[:, 0].values  # 节点ID
    timesteps = nodes_df.iloc[:, 1].values  # 时间步（可选：作为特征/过滤）
    feat_data = nodes_df.iloc[:, 2:].values  # 166维特征

    # 2. 加载标签（映射：1=欺诈，2=合法，0/unknown=未知）
    class_map = {"1": 1, "2": 0, "unknown": -1}
    classes_df["label"] = classes_df["class"].map(class_map)
    label_dict = dict(zip(classes_df["txId"], classes_df["label"]))
    labels = np.array([label_dict[node_id] for node_id in node_ids])

    # 3. 加载边（txId1 -> txId2）
    edges_df = pd.read_csv("elliptic_txs_edgelist.csv")
    edges = edges_df[["txId1", "txId2"]].values

    # 4. 过滤未知标签的节点（只保留欺诈/合法）
    valid_mask = labels != -1
    valid_node_ids = node_ids[valid_mask]
    valid_feats = feat_data[valid_mask]
    valid_labels = labels[valid_mask]

    # 5. 映射节点ID到连续索引（适配PyG）
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(valid_node_ids)}
    # 过滤边：只保留两端都是有效节点的边
    valid_edges = []
    for src, dst in edges:
        if src in node_id_to_idx and dst in node_id_to_idx:
            valid_edges.append([node_id_to_idx[src], node_id_to_idx[dst]])
    valid_edges = np.array(valid_edges)

    logging.info(f"Elliptic数据集加载完成：")
    logging.info(
        f"  有效节点数：{len(valid_node_ids)} (欺诈={np.sum(valid_labels == 1)}, 合法={np.sum(valid_labels == 0)})")
    logging.info(f"  有效边数：{len(valid_edges)}")
    logging.info(f"  特征维度：{valid_feats.shape[1]}")

    return valid_feats, valid_edges, valid_labels


# ===================== 3. 数据预处理函数 =====================
def preprocess_elliptic_data(feat_data, edges, labels):
    """
    预处理数据：归一化特征、构建PyG格式边索引、划分数据集
    :return: features (torch.FloatTensor), edge_index (torch.LongTensor), edge_type (torch.LongTensor),
             labels (np.ndarray), idx_train, idx_val, idx_test
    """
    # 1. 特征归一化（标准化：均值0，方差1）
    scaler = StandardScaler()
    feat_data = scaler.fit_transform(feat_data)

    # 2. 构建PyG格式边索引（[2, num_edges]）
    # Elliptic是无向边，双向添加（可选：保留单向）
    src = edges[:, 0]
    dst = edges[:, 1]
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_index = torch.LongTensor(edge_index)

    # 3. 边类型：Elliptic无多关系，统一标记为0
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)

    # 4. 数据集划分（分层抽样，保证类别平衡）
    index = np.arange(len(labels))
    idx_train_val, idx_test = train_test_split(
        index, stratify=labels, test_size=0.2, random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_train_val, stratify=labels[idx_train_val], test_size=0.125, random_state=seed
    )

    # 5. 转换特征为Tensor（归一化后）
    features = torch.FloatTensor(feat_data)

    print(f"数据预处理完成：")
    print(f"  训练集节点数：{len(idx_train)}")
    print(f"  验证集节点数：{len(idx_val)}")
    print(f"  测试集节点数：{len(idx_test)}")
    print(f"  边索引形状：{edge_index.shape}")

    return features, edge_index, edge_type, labels, idx_train, idx_val, idx_test


if __name__ == "__main__":

    # 2. 加载并预处理数据
    feat_data, edges, labels = load_elliptic_data()
    features, edge_index, edge_type, labels, idx_train, idx_val, idx_test = preprocess_elliptic_data(
        feat_data, edges, labels
    )

    # 3. 移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # 4. 输出适配后的格式（与你的RGCN模型输入对齐）
    print(f"\n适配RGCN模型的输入格式：")
    print(f"  features: {type(features)}, shape={features.shape}, device={features.device}")
    print(f"  edge_index: {type(edge_index)}, shape={edge_index.shape}, device={edge_index.device}")
    print(f"  edge_type: {type(edge_type)}, shape={edge_type.shape}, device={edge_type.device}")
    print(f"  labels: 欺诈={np.sum(labels == 1)}, 合法={np.sum(labels == 0)}")
    print(f"  训练集/验证集/测试集：{len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
