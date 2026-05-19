import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

log_text = '''
2025-11-19 10:47:08,539 - INFO - Epoch: 0, loss: 0.23109620809555054, time: 0.03472781181335449s
2025-11-19 10:47:08,602 - INFO - Macro F1: 0.1268
2025-11-19 10:47:08,602 - INFO - Accuracy: 0.1453
2025-11-19 10:47:08,602 - INFO - auc: 0.4846
2025-11-19 10:47:08,614 - INFO - Epoch: 1, loss: 0.23105619847774506, time: 0.005017518997192383s
2025-11-19 10:47:08,625 - INFO - Epoch: 2, loss: 0.23098056018352509, time: 0.004555702209472656s
2025-11-19 10:47:08,637 - INFO - Epoch: 3, loss: 0.23097620904445648, time: 0.005035877227783203s
2025-11-19 10:47:08,654 - INFO - Epoch: 4, loss: 0.2313917875289917, time: 0.00604701042175293s
2025-11-19 10:47:08,666 - INFO - Epoch: 5, loss: 0.2313273400068283, time: 0.005016326904296875s
2025-11-19 10:47:08,679 - INFO - Epoch: 6, loss: 0.23100222647190094, time: 0.005012989044189453s
2025-11-19 10:47:08,692 - INFO - Epoch: 7, loss: 0.23116230964660645, time: 0.00621485710144043s
2025-11-19 10:47:08,705 - INFO - Epoch: 8, loss: 0.2312464714050293, time: 0.004514932632446289s
2025-11-19 10:47:08,716 - INFO - Epoch: 9, loss: 0.23122964799404144, time: 0.004008769989013672s
2025-11-19 10:47:08,727 - INFO - Epoch: 10, loss: 0.23115497827529907, time: 0.004024505615234375s
2025-11-19 10:47:08,788 - INFO - Macro F1: 0.4698
2025-11-19 10:47:08,788 - INFO - Accuracy: 0.8485
2025-11-19 10:47:08,790 - INFO - auc: 0.5180
2025-11-19 10:47:08,809 - INFO - Epoch: 11, loss: 0.23082131147384644, time: 0.005584001541137695s
2025-11-19 10:47:08,824 - INFO - Epoch: 12, loss: 0.2311464548110962, time: 0.006513357162475586s
2025-11-19 10:47:08,842 - INFO - Epoch: 13, loss: 0.2309669703245163, time: 0.005517005920410156s
2025-11-19 10:47:08,859 - INFO - Epoch: 14, loss: 0.23122860491275787, time: 0.0050127506256103516s
2025-11-19 10:47:08,871 - INFO - Epoch: 15, loss: 0.23108704388141632, time: 0.005517482757568359s
2025-11-19 10:47:08,888 - INFO - Epoch: 16, loss: 0.23101234436035156, time: 0.006626605987548828s
2025-11-19 10:47:08,906 - INFO - Epoch: 17, loss: 0.23092444241046906, time: 0.005630016326904297s
2025-11-19 10:47:08,923 - INFO - Epoch: 18, loss: 0.23061925172805786, time: 0.006173849105834961s
2025-11-19 10:47:08,941 - INFO - Epoch: 19, loss: 0.2307741641998291, time: 0.005013227462768555s
2025-11-19 10:47:08,953 - INFO - Epoch: 20, loss: 0.2310829907655716, time: 0.005532264709472656s
2025-11-19 10:47:09,012 - INFO - Macro F1: 0.5041
2025-11-19 10:47:09,012 - INFO - Accuracy: 0.6375
2025-11-19 10:47:09,012 - INFO - auc: 0.5507
2025-11-19 10:47:09,030 - INFO - Epoch: 21, loss: 0.2310156226158142, time: 0.007083415985107422s
2025-11-19 10:47:09,042 - INFO - Epoch: 22, loss: 0.23082874715328217, time: 0.006032228469848633s
2025-11-19 10:47:09,055 - INFO - Epoch: 23, loss: 0.23096346855163574, time: 0.004993438720703125s
2025-11-19 10:47:09,068 - INFO - Epoch: 24, loss: 0.23101000487804413, time: 0.006009817123413086s
2025-11-19 10:47:09,081 - INFO - Epoch: 25, loss: 0.23103439807891846, time: 0.004586935043334961s
2025-11-19 10:47:09,094 - INFO - Epoch: 26, loss: 0.23106493055820465, time: 0.005444765090942383s
2025-11-19 10:47:09,106 - INFO - Epoch: 27, loss: 0.23083637654781342, time: 0.004463911056518555s
2025-11-19 10:47:09,118 - INFO - Epoch: 28, loss: 0.23088864982128143, time: 0.005448818206787109s
2025-11-19 10:47:09,129 - INFO - Epoch: 29, loss: 0.2309681624174118, time: 0.004007816314697266s
2025-11-19 10:47:09,141 - INFO - Epoch: 30, loss: 0.23072904348373413, time: 0.0045130252838134766s
2025-11-19 10:47:09,199 - INFO - Macro F1: 0.1989
2025-11-19 10:47:09,199 - INFO - Accuracy: 0.2038
2025-11-19 10:47:09,199 - INFO - auc: 0.5795
'''





# 解析日志数据
data = []
lines = log_text.strip().split('\n')
for line in lines:
    if 'Epoch' in line and 'loss' in line:
        # 提取Epoch、loss和各项指标
        parts = line.split(',')
        epoch = int(parts[1].split('Epoch: ')[1].strip())
        loss = float(parts[2].split(': ')[1].strip())
    elif 'Macro F1' in line:
        parts = line.split(',')
        macro_f1 = float(parts[1].split('Macro F1: ')[1].strip())
    elif 'Accuracy' in line:
        parts = line.split(',')
        accuracy = float(parts[1].split('Accuracy: ')[1].strip())
    elif 'auc' in line:
        parts = line.split(',')
        auc = float(parts[1].split('auc: ')[1].strip())

        data.append({
            'Epoch': epoch,
            'Loss': loss,
            'Macro F1': macro_f1,
            'Accuracy': accuracy,
            'AUC': auc
        })

# 转换为DataFrame
df = pd.DataFrame(data)

# 绘制训练过程图表
plt.figure(figsize=(12, 8))

# 子图1：Loss变化
plt.subplot(2, 2, 1)
plt.plot(df['Epoch'], df['Loss'], marker='o', color='blue')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 子图2：F1变化
plt.subplot(2, 2, 2)
plt.plot(df['Epoch'], df['Macro F1'], marker='o', color='green')
plt.title('Macro F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Macro F1')
plt.grid(True)

# 子图3：Precision变化
plt.subplot(2, 2, 3)
plt.plot(df['Epoch'], df['Accuracy'], marker='o', color='red')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# 子图4：AUC变化
plt.subplot(2, 2, 4)
plt.plot(df['Epoch'], df['AUC'], marker='o', color='purple')
plt.title('AUC per Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.grid(True)

plt.tight_layout()
plt.savefig('MLP_training.png')  # 保存为图片
plt.close()