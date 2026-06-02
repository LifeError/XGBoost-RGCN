# GNN-based Fraud Detection Research

基于图神经网络（GNN）的欺诈检测研究，对比多种 GNN 变体及集成方法在反欺诈数据集上的表现。

## 数据集

| 数据集 | 领域 | 说明 |
|---|---|---|
| YelpChi | 虚假评论检测 | 包含多种关系边（homo, net_rur, net_rtr, net_rsr） |
| Elliptic | 比特币交易欺诈 | 交易图分类 |
| IBM HI-LI | 反洗钱 | 银行交易网络 |

数据集文件较大（.mat / .csv），已通过 `.gitignore` 排除，需自行下载放置于对应目录。

## 模型

### 基准模型
| 模型 | 文件 | 说明 |
|---|---|---|
| MLP | `Test(MLP).py` | 多层感知机基线 |
| LSTM | `LSTM.py` | LSTM 序列模型 |
| XGBoost | `XGBoost.py` | 传统机器学习基线 |
| GCN | `Test(GCN).py` | 图卷积网络 |
| GAT | `Test(GAT).py` | 图注意力网络 |
| GAE | `Test(GAE).py` | 图自编码器 |

### 进阶模型
| 模型 | 文件 | 说明 |
|---|---|---|
| RGCN | `Test(RGCN).py` | 关系图卷积网络 |
| ResGCN | `Test(ResGCN).py` | 残差图卷积网络 |
| Self-Attention GNN | `Self-Attention-GNN/model.py` | 自注意力图神经网络 |

### 集成模型
| 模型 | 文件 | 说明 |
|---|---|---|
| RGCN + RandomForest | `RGCN-Forest.py` / `Test17-RGCN-RF.py` | RGCN 特征提取 + 随机森林分类器 |
| RGCN + XGBoost | `RGCN-XGBoost.py` / `Tset13-RGCN-XGBoost_Resul.py` | RGCN 特征提取 + XGBoost 分类器 |
| ResGCN + RandomForest | `ResGCN-RF.py` | ResGCN 特征提取 + 随机森林 |
| ResGCN + XGBoost | `ResGCN-XGBoost.py` | ResGCN 特征提取 + XGBoost |

### 训练脚本
| 文件 | 说明 |
|---|---|
| `Test9-GAT_training.py` | GAT 训练流程 |
| `Test10-GAE_training.py` | GAE 训练流程 |
| `Test11-MLP_training.py` | MLP 训练流程 |
| `Test14-ResGCN_training.py` | ResGCN 训练流程 |

### 分析脚本
| 文件 | 说明 |
|---|---|
| `Test15-threshold_fpr_recall_bar.py` | 不同阈值下 FPR / Recall 对比 |
| `Test16-dropout_fpr_recall_bar.py` | 不同 Dropout 率下性能对比 |
| `Test19-xgboost_feature_enhancement.py` | XGBoost 特征增强分析 |

## 子目录

| 目录 | 内容 |
|---|---|
| `Self-Attention-GNN/` | 自注意力 GNN 模型定义与训练日志 |
| `Test_on_Elliptic/` | Elliptic 数据集上的 RGCN+XGBoost 实验 |
| `Test_on_IBM/` | IBM HI/LI 反洗钱数据集实验 |
| `combine/` | 多模型 ROC 曲线对比 |
| `work/` | YelpChi 数据探索、PCA 可视化、特征重要性分析 |

## 技术要点

- **混合精度训练**：使用 `torch.cuda.amp` 加速训练并降低显存
- **早停机制**：基于验证集 F1/Loss 的早停策略
- **学习率调度**：CosineAnnealingLR 余弦退火
- **多阈值评估**：在不同分类阈值下评估 F1、Precision、Recall、FPR、AUC
- **集成策略**：GNN 提取节点嵌入 → 传统 ML 分类器（RF / XGBoost）

## 依赖

详见 `requirements.txt`，一键安装：

```bash
pip install -r requirements.txt
```

## 注意
项目在 Elliptic 和 IBM HI-LI 上的测试部分是残缺的
