import sys
import numpy as np
import pandas as pd
import os
import hashlib
import scipy.sparse as sp
from scipy.io import loadmat
import warnings

warnings.filterwarnings('ignore')


def calculate_file_md5(file_path, block_size=2 ** 20):
    """计算文件MD5哈希值（验证文件完整性）"""
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


def save_data_to_csv(data, output_dir="YelpChi_CSV_Export"):
    """
    将mat文件数据导出为CSV/TXT文件
    :param data: loadmat加载的字典
    :param output_dir: 输出目录
    """
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    readme_content = []  # 记录每个文件的说明
    readme_content.append("YelpChi.mat 数据导出说明\n")
    readme_content.append("=" * 50 + "\n")

    # 2. 遍历所有变量
    for key, value in data.items():
        print(f"正在导出: {key}...")

        # -------------------- 处理系统变量（非数值型，保存为TXT） --------------------
        if key in {'__header__', '__version__', '__globals__'}:
            file_path = os.path.join(output_dir, f"{key}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"变量名: {key}\n内容: {str(value)}")
            readme_content.append(f"{key}.txt: 系统元信息（mat文件头/版本/全局变量）\n")
            continue

        # -------------------- 处理一维数组（如label） --------------------
        if isinstance(value, np.ndarray) and value.ndim == 1:
            file_path = os.path.join(output_dir, f"{key}.csv")
            pd.DataFrame({key: value.flatten()}).to_csv(file_path, index=False, encoding='utf-8')
            readme_content.append(f"{key}.csv: 一维数组，形状={value.shape}\n")
            continue

        # -------------------- 处理二维数组 --------------------
        if isinstance(value, np.ndarray) and value.ndim == 2:
            file_path = os.path.join(output_dir, f"{key}.csv")
            # 生成列名（col_0, col_1...）
            cols = [f"col_{i}" for i in range(value.shape[1])]
            pd.DataFrame(value, columns=cols).to_csv(file_path, index=False, encoding='utf-8')
            readme_content.append(f"{key}.csv: 二维数组，形状={value.shape}\n")
            continue

        # -------------------- 处理稀疏矩阵（核心） --------------------
        if sp.issparse(value):
            # 稀疏矩阵信息记录
            shape_info = f"稀疏矩阵，形状={value.shape}，非零元素数={value.nnz}"
            readme_content.append(f"{key}_sparse.csv: {shape_info}（行索引+列索引+非零值）\n")

            # 转换为COO格式（方便提取行列索引）
            value_coo = value.tocoo()
            # 方式1：小矩阵 → 保存稠密CSV（可选）
            if value.shape[0] * value.shape[1] <= 10000:  # 仅小矩阵转稠密
                dense_file = os.path.join(output_dir, f"{key}_dense.csv")
                pd.DataFrame(value.todense()).to_csv(dense_file, index=False, encoding='utf-8')
                readme_content.append(f"{key}_dense.csv: {shape_info}（稠密矩阵版本）\n")

            # 方式2：所有稀疏矩阵 → 保存非零元素的坐标+值（通用）
            sparse_df = pd.DataFrame({
                "row_idx": value_coo.row,  # 非零元素行索引
                "col_idx": value_coo.col,  # 非零元素列索引
                "value": value_coo.data  # 非零元素值
            })
            sparse_file = os.path.join(output_dir, f"{key}_sparse.csv")
            sparse_df.to_csv(sparse_file, index=False, encoding='utf-8')
            continue

        # -------------------- 其他类型（兜底） --------------------
        file_path = os.path.join(output_dir, f"{key}_other.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"变量名: {key}\n类型: {type(value).__name__}\n内容: {str(value)}")
        readme_content.append(f"{key}_other.txt: 非数组/稀疏矩阵类型，类型={type(value).__name__}\n")

    # 3. 生成README文件（关键：说明每个文件含义）
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.writelines(readme_content)

    print(f"\n✅ 所有数据导出完成！")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"📖 请查看 {readme_path} 了解每个文件的含义")


def load_and_export_yelpchi(file_path="YelpChi.mat"):
    """加载数据并导出为CSV"""
    # 1. 校验文件
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {os.path.abspath(file_path)}")
        return

    # 2. 验证文件完整性
    file_md5 = calculate_file_md5(file_path)
    print(f"文件MD5: {file_md5}")

    # 3. 加载mat文件
    print(f"正在加载 {file_path}...")
    data = loadmat(file_path)
    print(f"✅ 数据加载成功，共包含 {len(data)} 个变量")

    # 4. 导出为CSV/TXT
    save_data_to_csv(data)


if __name__ == "__main__":
    # 主函数：执行加载+导出
    load_and_export_yelpchi()