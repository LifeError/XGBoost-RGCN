import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import hashlib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import networkx as nx
import warnings
import random

from sklearn.preprocessing import StandardScaler

# 忽略字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 设置英文字体
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 指定保存路径
SAVE_DIR = r"E:\Python\projectes\lunwen\work"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"图片将保存到: {SAVE_DIR}")


def calculate_file_md5(file_path, block_size=2 ** 20):
    """计算文件的MD5哈希值"""
    if not os.path.exists(file_path):
        return None

    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def load_and_inspect_aml_data(file_path='YelpChi.mat'):
    """加载并分析反洗钱数据集"""
    try:
        print(f"Loading anti-money laundering dataset: {file_path}")

        # 验证文件存在性和完整性
        if not os.path.exists(file_path):
            print(f"[错误] 文件不存在: {file_path}")
            return None

        # 计算文件哈希值
        file_md5 = calculate_file_md5(file_path)
        print(f"文件MD5哈希值: {file_md5}")

        # 尝试加载文件
        data = loadmat(file_path)
        print(f"Data loaded successfully, containing {len(data)} variables")

        # 显示所有变量名
        print("\nVariables in the dataset:")
        for i, key in enumerate(data.keys()):
            print(f"  {i + 1}. {key}")

        # 打印所有变量及其形状
        print("\n数据集详细内容:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
            elif hasattr(value, 'shape'):  # 处理稀疏矩阵
                print(f"  {key}: 形状={value.shape}, 类型={type(value).__name__}")
            else:
                print(f"  {key}: 类型={type(value).__name__}")

        return data

    except Exception as e:
        print(f"Error loading data: {e}")

        # 提供额外的故障排除建议
        print("\n故障排除建议:")
        print("1. 检查文件路径是否正确")
        print(f"   当前尝试路径: {os.path.abspath(file_path)}")
        print("2. 验证文件是否完整且未损坏")
        print("3. 尝试使用MATLAB打开文件确认其可访问性")

        return None


def visualize_aml_dataset(data):
    """可视化反洗钱数据集的关键特征"""
    if data is None:
        print("No data to visualize.")
        return

    generated_plots = []

    # 可视化特征矩阵
    if 'features' in data and 'label' in data:
        features = data['features']
        labels = data['label']

        # 转换label为一维数组
        if labels.ndim > 1 and (labels.shape[0] == 1 or labels.shape[1] == 1):
            print(f"  Reshaping labels from {labels.shape} to 1D array")
            labels = labels.flatten()

        print(f"  Labels shape after reshaping: {labels.shape}")

        if isinstance(features, np.ndarray) or hasattr(features, 'todense'):
            print("\nVisualizing feature matrix...")

            # PCA降维可视化
            if features.shape[1] >= 2:
                try:
                    print(f"  Features shape: {features.shape}")
                    print("  Performing PCA...")

                    # 转换为密集矩阵进行PCA
                    if hasattr(features, 'todense'):
                        print("  Converting sparse matrix to dense matrix for PCA...")
                        features_dense = features.todense()
                    else:
                        features_dense = features

                    # 检查是否有缺失值
                    if np.isnan(features_dense).any():
                        print("  Warning: Features contain NaN values. Filling with 0.")
                        features_dense = np.nan_to_num(features_dense)

                    # 转换为numpy数组
                    features_array = np.asarray(features_dense)
                    # 显示样本数据
                    print(f"  Sample data (first 5 rows, first 5 columns):")
                    print(pd.DataFrame(features_array[:5, :5]).to_string())

                    features_pca = PCA(n_components=2).fit_transform(features_array)
                    print(f"  PCA completed. Explained variance ratio: {PCA(n_components=2).fit(features_array).explained_variance_ratio_}")

                    print(features_array.shape)

                    suspicious_features = features_array[labels == 1]
                    print("可疑交易特征行（形状）：", suspicious_features.shape)
                    print("可疑交易特征示例：\n", suspicious_features[:5])  # 打印前5行

                    # 同理，筛选标签为0的正常交易特征行
                    normal_features = features_array[labels == 0]
                    print("正常交易特征行（形状）：", normal_features.shape)
                    print('正常交易特征示例：\n', normal_features[:5])

                    plt.figure(figsize=(12, 10))
                    plt.scatter(features_pca[labels == 0, 0], features_pca[labels == 0, 1],
                                c='blue', label='Normal Transactions', alpha=0.5, s=30)
                    plt.scatter(features_pca[labels == 1, 0], features_pca[labels == 1, 1],
                                c='red', label='Suspicious Transactions', alpha=0.8, s=50, edgecolors='black')

                    plt.title('PCA Visualization of Transaction Features', fontsize=16)
                    plt.xlabel('Principal Component 1', fontsize=14)
                    plt.ylabel('Principal Component 2', fontsize=14)
                    plt.legend(title='Transaction Type', fontsize=12, loc='upper right')
                    plt.grid(True, linestyle='--', alpha=0.7)

                    # 保存图片到指定路径
                    save_path = os.path.join(SAVE_DIR, 'pca_visualization.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"[成功] PCA visualization saved to: {save_path}")
                    generated_plots.append('pca_visualization.png')
                    plt.close()

                except Exception as e:
                    print(f"[错误] Error in PCA visualization: {e}")
                    import traceback
                    traceback.print_exc()  # 打印详细错误堆栈

            # 特征重要性评估
            if features.shape[1] > 0 and features.shape[0] > 100:
                try:
                    print("\nEvaluating feature importance...")
                    print(f"  Features shape: {features.shape}")
                    print(f"  Labels shape: {labels.shape}")
                    print(f"  Label distribution: {np.bincount(labels)}")

                    # 转换为密集矩阵进行随机森林
                    if hasattr(features, 'todense'):
                        print("  Converting sparse matrix to dense matrix for Random Forest...")
                        features_dense = features.todense()
                    else:
                        features_dense = features

                    # 检查是否有缺失值
                    if np.isnan(features_dense).any():
                        print("  Warning: Features contain NaN values. Filling with 0.")
                        features_dense = np.nan_to_num(features_dense)

                    # 转换为numpy数组
                    features_array = np.asarray(features_dense)

                    X_train, X_test, y_train, y_test = train_test_split(
                        features_array, labels, test_size=0.3, random_state=42, stratify=labels
                    )

                    print(f"  Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
                    print(f"  Training labels distribution: {np.bincount(y_train)}")
                    print(f"  Test labels distribution: {np.bincount(y_test)}")

                    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    print("  Training Random Forest model...")
                    rf.fit(X_train, y_train)

                    y_pred_proba = rf.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                    print(f"  Random Forest Model AUC: {auc:.4f}")

                    importances = rf.feature_importances_
                    indices = np.argsort(importances)  # 按重要性排序所有特征

                    plt.figure(figsize=(12, 16))  # 增加图片高度以容纳所有32个特征
                    plt.title('All 32 Feature Importance', fontsize=16)
                    plt.barh(range(len(indices)), importances[indices], align='center')
                    plt.yticks(range(len(indices)), indices)  # 使用特征索引作为y轴标签
                    plt.xlabel('Importance', fontsize=14)

                    # 保存图片到指定路径
                    save_path = os.path.join(SAVE_DIR, 'feature_importance_all32.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"[成功] Feature importance chart (All 32) saved to: {save_path}")
                    generated_plots.append('feature_importance_all32.png')
                    plt.close()

                except Exception as e:
                    print(f"[错误] Error in feature importance evaluation: {e}")
                    import traceback
                    traceback.print_exc()  # 打印详细错误堆栈
        else:
            print(f"[错误] Features 类型不支持: {type(features).__name__}")
    else:
        print("[错误] 缺少 features 或 label 变量，无法生成特征可视化图")

    # 可视化图结构
    graph_keys = ['homo', 'net_rur', 'net_rtr', 'net_rsr']
    for key in graph_keys:
        if key in data:
            adj_matrix = data[key]

            if isinstance(adj_matrix, np.ndarray) or hasattr(adj_matrix, 'todense'):
                try:
                    print(f"\nVisualizing transaction network graph: {key}")
                    print(f"  Adjacency matrix shape: {adj_matrix.shape}")

                    # 计算图密度
                    if hasattr(adj_matrix, 'nnz'):  # 稀疏矩阵
                        density = adj_matrix.nnz / (adj_matrix.shape[0] * adj_matrix.shape[1])
                    else:  # 密集矩阵
                        density = np.count_nonzero(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1])

                    print(f"  Graph density: {density:.10f}")

                    max_nodes = min(500, adj_matrix.shape[0])

                    G = nx.Graph()
                    # 转换为密集矩阵
                    if hasattr(adj_matrix, 'todense'):
                        print(f"  Converting sparse adjacency matrix to dense (top {max_nodes} nodes)...")
                        # 先转换为CSR格式以支持切片
                        adj_csr = adj_matrix.tocsr()
                        adj_dense = adj_csr[:max_nodes, :max_nodes].todense()
                    else:
                        adj_dense = adj_matrix[:max_nodes, :max_nodes]

                    # 构建图
                    print("  Building graph...")
                    edge_count = 0
                    for i in range(max_nodes):
                        for j in range(i + 1, max_nodes):
                            if adj_dense[i, j] != 0:
                                G.add_edge(i, j)
                                edge_count += 1

                    print(f"  Graph built with {G.number_of_nodes()} nodes and {edge_count} edges")

                    plt.figure(figsize=(15, 12))
                    print("  Calculating node positions (this may take time for large graphs)...")
                    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)


                    # all_keys = list(data.keys())
                    # print("data中的所有列名/变量名：")
                    # for idx, key in enumerate(all_keys, 1):
                    #     print(f"  {idx}. {key}")
                    # print("\n简化输出：", all_keys)
                    # if 'label' in data:
                    #     label_array = data['label']
                    #     if label_array.ndim == 2:
                    #         label_array = label_array.flatten()  # 扁平化：(1,45954) → (45954,)
                    #     print("label数组形状（扁平化后）：", label_array.shape)
                    #     print("label数组的唯一值：", np.unique(label_array))
                    #     print(G.number_of_nodes())
                    # else:
                    #     print('np label')



                    # 如果有标签，按标签着色



                    if 'label' in data and G.number_of_nodes() <= len(labels):
                        print('1')
                        labels_flat = labels.flatten()
                        sample_nodes = list(G.nodes())  # 抽样的500个节点索引
                        sample_labels = [labels_flat[i] for i in sample_nodes]
                        print(f"抽样节点数：{len(sample_nodes)}")
                        print(f"抽样节点中正常交易（0）数量：{sample_labels.count(0)}")
                        print(f"抽样节点中可疑交易（1）数量：{sample_labels.count(1)}")
                        print(f"抽样节点标签唯一值：{set(sample_labels)}")  # 验证标签值是否为0/1
                        # 绘制正常节点
                        nx.draw_networkx_nodes(G, pos,
                                               nodelist=[i for i in G.nodes() if labels[i] == 0],
                                               node_color='#3274A1', node_size=50, alpha=0.6,
                                              )

                        # 绘制可疑节点
                        nx.draw_networkx_nodes(G, pos,
                                               nodelist=[i for i in G.nodes() if labels[i] == 1],
                                               node_color='#E1812C', node_size=100, alpha=0.8,
                                               edgecolors='black', linewidths=1,)
                    else:
                        print('2')
                        nx.draw_networkx_nodes(G, pos, node_color='#3274A1', node_size=50, alpha=0.6)

                    # 绘制边
                    print("  Drawing edges...")
                    nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.5)

                    plt.title(f'{key}', fontsize=16)
                    plt.legend(fontsize=12)
                    plt.axis('off')

                    # 保存图片到指定路径
                    save_path = os.path.join(SAVE_DIR, f'network_{key}_new.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"[成功] Transaction network graph saved to: {save_path}")
                    generated_plots.append(f'network_{key}.png')
                    plt.close()

                except Exception as e:
                    print(f"[错误] Error in graph visualization: {e}")
                    import traceback
                    traceback.print_exc()  # 打印详细错误堆栈
            else:
                print(f"[错误] Adjacency matrix 类型不支持: {type(adj_matrix).__name__}")
        else:
            print(f"[错误] 数据集中不存在 {key} 变量，无法生成网络图")

    # 可视化标签分布
    if 'label' in data:
        labels = data['label']

        # 转换label为一维数组
        if labels.ndim > 1 and (labels.shape[0] == 1 or labels.shape[1] == 1):
            labels = labels.flatten()

        unique_labels, counts = np.unique(labels, return_counts=True)

        if len(counts) > 0:
            try:
                print("\nVisualizing label distribution...")
                print(f"  Label values: {unique_labels}")
                print(f"  Label counts: {counts}")

                plt.figure(figsize=(10, 6))
                bars = plt.bar(unique_labels, counts, color=['#3274A1', '#E1812C'])
                plt.title('Transaction Label Distribution', fontsize=16)
                plt.xlabel('Label Value (0=Normal, 1=Suspicious)', fontsize=14)
                plt.ylabel('Number of Transactions', fontsize=14)
                plt.xticks(unique_labels)

                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                             f'{height}', ha='center', va='bottom', fontsize=12)

                # 计算并显示比例
                if len(counts) > 1:
                    total = sum(counts)
                    plt.text(0, max(counts) * 0.9, f'{counts[0] / total:.2%}', ha='center', fontsize=14, color='black')
                    plt.text(1, max(counts) * 0.9, f'{counts[1] / total:.2%}',ha='center', fontsize=14, color='black')

                # 保存图片到指定路径
                save_path = os.path.join(SAVE_DIR, 'label_distribution.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[成功] Label distribution chart saved to: {save_path}")
                generated_plots.append('label_distribution.png')
                plt.close()
            except Exception as e:
                print(f"[错误] Error in label distribution visualization: {e}")
                import traceback
                traceback.print_exc()  # 打印详细错误堆栈
    else:
        print("[错误] 缺少 label 变量，无法生成标签分布图")

    # 打印生成的图片列表
    if generated_plots:
        print(f"\n[完成] 共生成 {len(generated_plots)} 张图片:")
        for plot in generated_plots:
            print(f"  - {plot}")
    else:
        print("[错误] 没有生成任何图片，请检查数据集和错误信息")


def main():
    """主函数"""
    # 加载并检查数据
    data = load_and_inspect_aml_data()

    if data:
        # 可视化选项
        visualize_option = input("\nVisualize the dataset? (y/n): ")
        if visualize_option.lower() == 'y':
            visualize_aml_dataset(data)
            print(f"\n所有可视化图表已保存到: {SAVE_DIR}")


if __name__ == "__main__":
    main()