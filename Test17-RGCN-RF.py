import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# 原日志内容处理
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

# 新的日志内容处理
new_log_text = '''
2025-11-19 10:14:09,716 - INFO - Epoch 0 | Loss: 38.1275 | F1: 0.5265 | Precision: 0.8575 | Recall: 0.5334 | FPR: 0.0020 | AUC: 0.8737 | Early Stopping Metric: 1.0648
2025-11-19 10:14:31,727 - INFO - Epoch 20 | Loss: 0.0329 | F1: 0.8037 | Precision: 0.8872 | Recall: 0.7577 | FPR: 0.0160 | AUC: 0.9290 | Early Stopping Metric: 1.4994
2025-11-19 10:14:53,679 - INFO - Epoch 40 | Loss: 0.0261 | F1: 0.8093 | Precision: 0.8871 | Recall: 0.7648 | FPR: 0.0168 | AUC: 0.9307 | Early Stopping Metric: 1.5128
2025-11-19 10:15:15,727 - INFO - Epoch 60 | Loss: 0.0206 | F1: 0.8106 | Precision: 0.8936 | Recall: 0.7642 | FPR: 0.0150 | AUC: 0.9315 | Early Stopping Metric: 1.5134
2025-11-19 10:15:37,695 - INFO - Epoch 80 | Loss: 0.0212 | F1: 0.7929 | Precision: 0.8979 | Recall: 0.7418 | FPR: 0.0120 | AUC: 0.9296 | Early Stopping Metric: 1.4716
2025-11-19 10:15:59,568 - INFO - GPU内存使用情况 - 分配: 424.02 MB, 缓存: 508.00 MB
2025-11-19 10:15:59,689 - INFO - Epoch 100 | Loss: 0.0200 | F1: 0.8012 | Precision: 0.8972 | Recall: 0.7517 | FPR: 0.0130 | AUC: 0.9309 | Early Stopping Metric: 1.4905
2025-11-19 10:16:21,840 - INFO - Epoch 120 | Loss: 0.0187 | F1: 0.8076 | Precision: 0.8950 | Recall: 0.7601 | FPR: 0.0143 | AUC: 0.9308 | Early Stopping Metric: 1.5059
2025-11-19 10:16:43,968 - INFO - Epoch 140 | Loss: 0.0179 | F1: 0.7985 | Precision: 0.8980 | Recall: 0.7483 | FPR: 0.0125 | AUC: 0.9308 | Early Stopping Metric: 1.4840
2025-11-19 10:17:05,966 - INFO - Epoch 160 | Loss: 0.0182 | F1: 0.8032 | Precision: 0.8942 | Recall: 0.7550 | FPR: 0.0140 | AUC: 0.9310 | Early Stopping Metric: 1.4959
2025-11-19 10:17:28,048 - INFO - Epoch 180 | Loss: 0.0198 | F1: 0.8115 | Precision: 0.8931 | Recall: 0.7656 | FPR: 0.0153 | AUC: 0.9301 | Early Stopping Metric: 1.5159
2025-11-19 10:17:49,932 - INFO - GPU内存使用情况 - 分配: 424.01 MB, 缓存: 508.00 MB
2025-11-19 10:17:50,035 - INFO - Epoch 200 | Loss: 0.0191 | F1: 0.8029 | Precision: 0.8969 | Recall: 0.7539 | FPR: 0.0132 | AUC: 0.9308 | Early Stopping Metric: 1.4945
2025-11-19 10:18:12,110 - INFO - Epoch 220 | Loss: 0.0173 | F1: 0.8090 | Precision: 0.8956 | Recall: 0.7616 | FPR: 0.0143 | AUC: 0.9305 | Early Stopping Metric: 1.5089
'''
# 2025-11-19 10:18:34,625 - INFO - Epoch 240 | Loss: 0.0177 | F1: 0.8101 | Precision: 0.8908 | Recall: 0.7646 | FPR: 0.0158 | AUC: 0.9315 | Early Stopping Metric: 1.5133
# 2025-11-19 10:18:55,857 - INFO - Epoch 260 | Loss: 0.0181 | F1: 0.8073 | Precision: 0.8958 | Recall: 0.7595 | FPR: 0.0140 | AUC: 0.9307 | Early Stopping Metric: 1.5049
# 2025-11-19 10:19:17,205 - INFO - Epoch 280 | Loss: 0.0177 | F1: 0.8109 | Precision: 0.8946 | Recall: 0.7643 | FPR: 0.0148 | AUC: 0.9308 | Early Stopping Metric: 1.5139
# 2025-11-19 10:19:38,584 - INFO - GPU内存使用情况 - 分配: 424.02 MB, 缓存: 508.00 MB
# 2025-11-19 10:19:38,689 - INFO - Epoch 300 | Loss: 0.0207 | F1: 0.8336 | Precision: 0.8838 | Recall: 0.7991 | FPR: 0.0216 | AUC: 0.9309 | Early Stopping Metric: 1.5765
# 2025-11-19 10:19:59,756 - INFO - Epoch 320 | Loss: 0.0179 | F1: 0.8124 | Precision: 0.8909 | Recall: 0.7674 | FPR: 0.0160 | AUC: 0.9317 | Early Stopping Metric: 1.5188
# 2025-11-19 10:20:21,584 - INFO - Epoch 340 | Loss: 0.0178 | F1: 0.8118 | Precision: 0.8924 | Recall: 0.7662 | FPR: 0.0155 | AUC: 0.9312 | Early Stopping Metric: 1.5168
# 2025-11-19 10:20:44,312 - INFO - Epoch 360 | Loss: 0.0166 | F1: 0.7982 | Precision: 0.8989 | Recall: 0.7476 | FPR: 0.0122 | AUC: 0.9302 | Early Stopping Metric: 1.4830
# 2025-11-19 10:21:08,528 - INFO - Epoch 380 | Loss: 0.0172 | F1: 0.8061 | Precision: 0.8990 | Recall: 0.7570 | FPR: 0.0130 | AUC: 0.9307 | Early Stopping Metric: 1.5010
# 2025-11-19 10:21:30,057 - INFO - GPU内存使用情况 - 分配: 424.01 MB, 缓存: 508.00 MB
# 2025-11-19 10:21:30,162 - INFO - Epoch 400 | Loss: 0.0191 | F1: 0.8184 | Precision: 0.8824 | Recall: 0.7784 | FPR: 0.0196 | AUC: 0.9294 | Early Stopping Metric: 1.5371
# 2025-11-19 10:21:51,328 - INFO - Epoch 420 | Loss: 0.0182 | F1: 0.8159 | Precision: 0.8883 | Recall: 0.7728 | FPR: 0.0173 | AUC: 0.9318 | Early Stopping Metric: 1.5283
# 2025-11-19 10:22:15,475 - INFO - Epoch 440 | Loss: 0.0182 | F1: 0.8151 | Precision: 0.8863 | Recall: 0.7725 | FPR: 0.0178 | AUC: 0.9318 | Early Stopping Metric: 1.5272
# 2025-11-19 10:22:37,888 - INFO - Epoch 460 | Loss: 0.0179 | F1: 0.8121 | Precision: 0.8916 | Recall: 0.7668 | FPR: 0.0158 | AUC: 0.9315 | Early Stopping Metric: 1.5178
# 2025-11-19 10:23:00,391 - INFO - Epoch 480 | Loss: 0.0178 | F1: 0.8057 | Precision: 0.8980 | Recall: 0.7569 | FPR: 0.0132 | AUC: 0.9310 | Early Stopping Metric: 1.5005
# 2025-11-19 10:23:22,328 - INFO - GPU内存使用情况 - 分配: 424.01 MB, 缓存: 508.00 MB
# 2025-11-19 10:23:22,433 - INFO - Epoch 500 | Loss: 0.0179 | F1: 0.8101 | Precision: 0.8908 | Recall: 0.7646 | FPR: 0.0158 | AUC: 0.9305 | Early Stopping Metric: 1.5133

# 解析原日志数据
def parse_log(log_text):
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
    return pd.DataFrame(data)

df1 = parse_log(log_text)
df2 = parse_log(new_log_text)

# 绘制训练过程图表
plt.figure(figsize=(12, 8))

# 子图1：FPR变化
plt.subplot(2, 2, 1)
plt.plot(df1['Epoch'], df1['FPR'], marker='o', color='blue', label='RGCN-XGB')
plt.plot(df2['Epoch'], df2['FPR'], marker='s', color='red', label='RGCN-RF')
plt.title('False Positive Rate (FPR) per Epoch')
plt.xlabel('Epoch')
plt.ylabel('FPR')
plt.grid(True)
plt.legend()

# 子图2：F1变化
plt.subplot(2, 2, 2)
plt.plot(df1['Epoch'], df1['F1'], marker='o', color='blue', label='RGCN-XGB')
plt.plot(df2['Epoch'], df2['F1'], marker='s', color='red', label='RGCN-RF')
plt.title('F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.grid(True)
plt.legend()

# 子图3：Recall变化
plt.subplot(2, 2, 3)
plt.plot(df1['Epoch'], df1['Recall'], marker='o', color='blue', label='RGCN-XGB')
plt.plot(df2['Epoch'], df2['Recall'], marker='s', color='red', label='RGCN-RF')
plt.title('Recall per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.grid(True)
plt.legend()

# 子图4：AUC变化
plt.subplot(2, 2, 4)
plt.plot(df1['Epoch'], df1['AUC'], marker='o', color='blue', label='RGCN-XGB')
plt.plot(df2['Epoch'], df2['AUC'], marker='s', color='red', label='RGCN-RF')
plt.title('AUC per Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('RGCN-RF.png')  # 保存为图片
plt.close()