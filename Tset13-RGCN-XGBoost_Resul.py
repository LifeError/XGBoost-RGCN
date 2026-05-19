import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# 日志内容处理（假设已复制日志内容到log_text变量）
log_text = '''
2025-11-19 10:24:38,702 - INFO - Epoch 0 | Loss: 22.1087 | F1: 0.4608 | Precision: 0.9273 | Recall: 0.5000 | FPR: 0.0000 | AUC: 0.8118 | Early Stopping Metric: 1.0000
2025-11-19 10:24:50,616 - INFO - Epoch 20 | Loss: 2.9817 | F1: 0.8791 | Precision: 0.8820 | Recall: 0.8763 | FPR: 0.0334 | AUC: 0.9435 | Early Stopping Metric: 1.7192
2025-11-19 10:25:02,573 - INFO - Epoch 40 | Loss: 2.6573 | F1: 0.8771 | Precision: 0.8780 | Recall: 0.8761 | FPR: 0.0351 | AUC: 0.9425 | Early Stopping Metric: 1.7172
2025-11-19 10:25:14,765 - INFO - Epoch 60 | Loss: 2.6707 | F1: 0.8770 | Precision: 0.8809 | Recall: 0.8733 | FPR: 0.0334 | AUC: 0.9421 | Early Stopping Metric: 1.7132
2025-11-19 10:25:26,816 - INFO - Epoch 80 | Loss: 2.5212 | F1: 0.8778 | Precision: 0.8835 | Recall: 0.8724 | FPR: 0.0321 | AUC: 0.9431 | Early Stopping Metric: 1.7128
2025-11-19 10:25:38,832 - INFO - GPU内存使用情况 - 分配: 384.78 MB, 缓存: 460.00 MB
2025-11-19 10:25:38,898 - INFO - Epoch 100 | Loss: 2.5722 | F1: 0.8744 | Precision: 0.8742 | Recall: 0.8746 | FPR: 0.0367 | AUC: 0.9433 | Early Stopping Metric: 1.7126
2025-11-19 10:25:51,157 - INFO - Epoch 120 | Loss: 2.4548 | F1: 0.8767 | Precision: 0.8774 | Recall: 0.8760 | FPR: 0.0354 | AUC: 0.9449 | Early Stopping Metric: 1.7167
2025-11-19 10:26:03,667 - INFO - Epoch 140 | Loss: 2.4440 | F1: 0.8757 | Precision: 0.8769 | Recall: 0.8745 | FPR: 0.0354 | AUC: 0.9449 | Early Stopping Metric: 1.7137
2025-11-19 10:26:16,022 - INFO - Epoch 160 | Loss: 2.3558 | F1: 0.8737 | Precision: 0.8704 | Recall: 0.8772 | FPR: 0.0390 | AUC: 0.9450 | Early Stopping Metric: 1.7155
2025-11-19 10:26:28,327 - INFO - Epoch 180 | Loss: 2.3181 | F1: 0.8733 | Precision: 0.8724 | Recall: 0.8743 | FPR: 0.0374 | AUC: 0.9456 | Early Stopping Metric: 1.7111
2025-11-19 10:26:40,408 - INFO - GPU内存使用情况 - 分配: 384.78 MB, 缓存: 460.00 MB
2025-11-19 10:26:40,489 - INFO - Epoch 200 | Loss: 2.3438 | F1: 0.8733 | Precision: 0.8751 | Recall: 0.8714 | FPR: 0.0356 | AUC: 0.9463 | Early Stopping Metric: 1.7072
2025-11-19 10:26:52,524 - INFO - Epoch 220 | Loss: 2.3404 | F1: 0.8742 | Precision: 0.8721 | Recall: 0.8762 | FPR: 0.0379 | AUC: 0.9461 | Early Stopping Metric: 1.7146
'''

# 解析日志数据
data = []
lines = log_text.strip().split('\n')
for line in lines:
    if 'Epoch' in line and 'Loss' in line:
        # 提取Epoch、loss和各项指标
        parts = line.split('|')
        epoch = int(parts[0].split('Epoch ')[1].strip())
        loss = float(parts[1].split(': ')[1].strip())
        f1 = float(parts[2].split(': ')[1].strip())
        precision = float(parts[3].split(': ')[1].strip())
        recall = float(parts[4].split(': ')[1].strip())
        fpr = float(parts[5].split(': ')[1].strip())
        auc = float(parts[6].split(': ')[1].strip())

        data.append({
            'Epoch': epoch,
            'Loss': loss,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'FPR': fpr,
            'AUC': auc
        })

# 转换为DataFrame
df = pd.DataFrame(data)

# 绘制训练过程图表
plt.figure(figsize=(12, 8))

# 子图1：FPR变化
plt.subplot(2, 2, 1)
plt.plot(df['Epoch'], df['FPR'], marker='o', color='blue')
plt.title('False Positive Rate (FPR) per Epoch')
plt.xlabel('Epoch')
plt.ylabel('FPR')
plt.grid(True)

# 子图2：F1变化
plt.subplot(2, 2, 2)
plt.plot(df['Epoch'], df['F1'], marker='o', color='green')
plt.title('F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.grid(True)

# 子图3：Recall变化
plt.subplot(2, 2, 3)
plt.plot(df['Epoch'], df['Recall'], marker='o', color='red')
plt.title('Recall per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.grid(True)

# 子图4：AUC变化
plt.subplot(2, 2, 4)
plt.plot(df['Epoch'], df['AUC'], marker='o', color='purple')
plt.title('AUC per Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.grid(True)

plt.tight_layout()
plt.savefig('RGCN-XGBoost_Resul.png')  # 保存为图片
plt.close()