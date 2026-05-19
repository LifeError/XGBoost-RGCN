import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

log_text = '''
2025-11-19 10:27:22,632 - INFO - Epoch 0 | Loss: 1.3858 | F1: 0.4169 | Precision: 0.4941 | Recall: 0.4882 | FPR: 0.5161 | AUC: 0.4808
2025-11-19 10:27:28,043 - INFO - Epoch 20 | Loss: 1.2545 | F1: 0.4410 | Precision: 0.5060 | Recall: 0.5121 | FPR: 0.4789 | AUC: 0.5183
2025-11-19 10:27:33,488 - INFO - Epoch 40 | Loss: 1.1738 | F1: 0.4226 | Precision: 0.5064 | Recall: 0.5128 | FPR: 0.5310 | AUC: 0.5160
2025-11-19 10:27:38,824 - INFO - Epoch 60 | Loss: 1.1434 | F1: 0.3920 | Precision: 0.5050 | Recall: 0.5096 | FPR: 0.6032 | AUC: 0.5104
2025-11-19 10:27:44,289 - INFO - Epoch 80 | Loss: 1.1290 | F1: 0.3933 | Precision: 0.5023 | Recall: 0.5045 | FPR: 0.5953 | AUC: 0.5024
2025-11-19 10:27:49,721 - INFO - Epoch 100 | Loss: 1.1212 | F1: 0.4386 | Precision: 0.4960 | Recall: 0.4921 | FPR: 0.4562 | AUC: 0.4984
2025-11-19 10:27:55,266 - INFO - Epoch 120 | Loss: 1.0992 | F1: 0.4158 | Precision: 0.4999 | Recall: 0.4998 | FPR: 0.5338 | AUC: 0.5020
2025-11-19 10:28:00,759 - INFO - Epoch 140 | Loss: 1.0696 | F1: 0.4424 | Precision: 0.5024 | Recall: 0.5049 | FPR: 0.4643 | AUC: 0.5061
2025-11-19 10:28:06,215 - INFO - Epoch 160 | Loss: 1.0495 | F1: 0.4385 | Precision: 0.5034 | Recall: 0.5069 | FPR: 0.4788 | AUC: 0.5060
2025-11-19 10:28:11,765 - INFO - Epoch 180 | Loss: 1.0480 | F1: 0.4384 | Precision: 0.5059 | Recall: 0.5118 | FPR: 0.4861 | AUC: 0.5159
2025-11-19 10:28:17,356 - INFO - Epoch 200 | Loss: 1.0172 | F1: 0.4322 | Precision: 0.5079 | Recall: 0.5160 | FPR: 0.5090 | AUC: 0.5214
2025-11-19 10:28:22,937 - INFO - Epoch 220 | Loss: 1.0057 | F1: 0.4172 | Precision: 0.5022 | Recall: 0.5044 | FPR: 0.5355 | AUC: 0.5067
2025-11-19 10:28:28,552 - INFO - Epoch 240 | Loss: 0.9973 | F1: 0.4091 | Precision: 0.5003 | Recall: 0.5005 | FPR: 0.5521 | AUC: 0.5038
2025-11-19 10:28:34,087 - INFO - Epoch 260 | Loss: 1.0009 | F1: 0.4066 | Precision: 0.5049 | Recall: 0.5096 | FPR: 0.5683 | AUC: 0.5124
2025-11-19 10:28:39,529 - INFO - Epoch 280 | Loss: 0.9982 | F1: 0.4040 | Precision: 0.5023 | Recall: 0.5045 | FPR: 0.5694 | AUC: 0.5051
2025-11-19 10:28:45,087 - INFO - Epoch 300 | Loss: 0.9745 | F1: 0.4044 | Precision: 0.5020 | Recall: 0.5040 | FPR: 0.5679 | AUC: 0.5092
2025-11-19 10:28:50,589 - INFO - Epoch 320 | Loss: 1.0088 | F1: 0.4075 | Precision: 0.5045 | Recall: 0.5089 | FPR: 0.5655 | AUC: 0.5085
2025-11-19 10:28:56,137 - INFO - Epoch 340 | Loss: 0.9751 | F1: 0.4093 | Precision: 0.5028 | Recall: 0.5056 | FPR: 0.5574 | AUC: 0.5088
2025-11-19 10:29:01,640 - INFO - Epoch 360 | Loss: 0.9892 | F1: 0.4130 | Precision: 0.5030 | Recall: 0.5061 | FPR: 0.5483 | AUC: 0.5096
2025-11-19 10:29:07,185 - INFO - Epoch 380 | Loss: 0.9866 | F1: 0.4171 | Precision: 0.5091 | Recall: 0.5182 | FPR: 0.5513 | AUC: 0.5230
2025-11-19 10:29:12,687 - INFO - Epoch 400 | Loss: 0.9813 | F1: 0.4098 | Precision: 0.5043 | Recall: 0.5085 | FPR: 0.5591 | AUC: 0.5130
2025-11-19 10:29:18,270 - INFO - Epoch 420 | Loss: 0.9747 | F1: 0.4085 | Precision: 0.5020 | Recall: 0.5039 | FPR: 0.5574 | AUC: 0.5102
2025-11-19 10:29:23,814 - INFO - Epoch 440 | Loss: 0.9856 | F1: 0.4112 | Precision: 0.5061 | Recall: 0.5120 | FPR: 0.5593 | AUC: 0.5181
2025-11-19 10:29:29,315 - INFO - Epoch 460 | Loss: 0.9700 | F1: 0.4121 | Precision: 0.5038 | Recall: 0.5075 | FPR: 0.5522 | AUC: 0.5072
2025-11-19 10:29:35,003 - INFO - Epoch 480 | Loss: 0.9559 | F1: 0.4087 | Precision: 0.5049 | Recall: 0.5097 | FPR: 0.5632 | AUC: 0.5113
2025-11-19 10:29:40,635 - INFO - Epoch 500 | Loss: 0.9664 | F1: 0.4142 | Precision: 0.5065 | Recall: 0.5130 | FPR: 0.5528 | AUC: 0.5145
2025-11-19 10:29:46,304 - INFO - Epoch 520 | Loss: 0.9695 | F1: 0.4104 | Precision: 0.5046 | Recall: 0.5091 | FPR: 0.5584 | AUC: 0.5137
2025-11-19 10:29:51,746 - INFO - Epoch 540 | Loss: 0.9805 | F1: 0.4068 | Precision: 0.5045 | Recall: 0.5088 | FPR: 0.5670 | AUC: 0.5122
2025-11-19 10:29:57,289 - INFO - Epoch 560 | Loss: 0.9733 | F1: 0.4060 | Precision: 0.5037 | Recall: 0.5074 | FPR: 0.5674 | AUC: 0.5077
2025-11-19 10:30:02,840 - INFO - Epoch 580 | Loss: 0.9714 | F1: 0.4021 | Precision: 0.5036 | Recall: 0.5071 | FPR: 0.5767 | AUC: 0.5080
2025-11-19 10:30:08,431 - INFO - Epoch 600 | Loss: 0.9512 | F1: 0.4110 | Precision: 0.5029 | Recall: 0.5058 | FPR: 0.5531 | AUC: 0.5040
2025-11-19 10:30:14,476 - INFO - Epoch 620 | Loss: 0.9585 | F1: 0.4058 | Precision: 0.5043 | Recall: 0.5086 | FPR: 0.5692 | AUC: 0.5116
2025-11-19 10:30:20,059 - INFO - Epoch 640 | Loss: 0.9633 | F1: 0.4151 | Precision: 0.5033 | Recall: 0.5065 | FPR: 0.5434 | AUC: 0.5058
2025-11-19 10:30:25,519 - INFO - Epoch 660 | Loss: 0.9621 | F1: 0.4010 | Precision: 0.5025 | Recall: 0.5050 | FPR: 0.5772 | AUC: 0.5072
2025-11-19 10:30:30,980 - INFO - Epoch 680 | Loss: 0.9460 | F1: 0.4199 | Precision: 0.5047 | Recall: 0.5093 | FPR: 0.5342 | AUC: 0.5118
2025-11-19 10:30:36,555 - INFO - Epoch 700 | Loss: 0.9622 | F1: 0.4372 | Precision: 0.5055 | Recall: 0.5110 | FPR: 0.4884 | AUC: 0.5180
2025-11-19 10:30:41,982 - INFO - Epoch 720 | Loss: 0.9415 | F1: 0.4220 | Precision: 0.5034 | Recall: 0.5067 | FPR: 0.5255 | AUC: 0.5074
2025-11-19 10:30:47,727 - INFO - Epoch 740 | Loss: 0.9570 | F1: 0.4247 | Precision: 0.5028 | Recall: 0.5057 | FPR: 0.5169 | AUC: 0.5105
2025-11-19 10:30:53,922 - INFO - Epoch 760 | Loss: 0.9594 | F1: 0.4279 | Precision: 0.5040 | Recall: 0.5081 | FPR: 0.5109 | AUC: 0.5123
2025-11-19 10:31:00,117 - INFO - Epoch 780 | Loss: 0.9430 | F1: 0.4311 | Precision: 0.5051 | Recall: 0.5103 | FPR: 0.5049 | AUC: 0.5110
2025-11-19 10:31:06,309 - INFO - Epoch 800 | Loss: 0.9403 | F1: 0.4283 | Precision: 0.5037 | Recall: 0.5075 | FPR: 0.5092 | AUC: 0.5089
2025-11-19 10:31:12,465 - INFO - Epoch 820 | Loss: 0.9397 | F1: 0.4257 | Precision: 0.5043 | Recall: 0.5086 | FPR: 0.5175 | AUC: 0.5134
2025-11-19 10:31:18,691 - INFO - Epoch 840 | Loss: 0.9467 | F1: 0.4232 | Precision: 0.5006 | Recall: 0.5012 | FPR: 0.5153 | AUC: 0.5019
2025-11-19 10:31:24,807 - INFO - Epoch 860 | Loss: 0.9530 | F1: 0.4187 | Precision: 0.5034 | Recall: 0.5067 | FPR: 0.5342 | AUC: 0.5087
2025-11-19 10:31:30,954 - INFO - Epoch 880 | Loss: 0.9565 | F1: 0.4150 | Precision: 0.5022 | Recall: 0.5043 | FPR: 0.5413 | AUC: 0.5085
2025-11-19 10:31:37,141 - INFO - Epoch 900 | Loss: 0.9564 | F1: 0.4163 | Precision: 0.5046 | Recall: 0.5091 | FPR: 0.5432 | AUC: 0.5075
2025-11-19 10:31:43,501 - INFO - Epoch 920 | Loss: 0.9281 | F1: 0.4213 | Precision: 0.5006 | Recall: 0.5012 | FPR: 0.5204 | AUC: 0.5052
2025-11-19 10:31:50,181 - INFO - Epoch 940 | Loss: 0.9426 | F1: 0.4144 | Precision: 0.5033 | Recall: 0.5066 | FPR: 0.5453 | AUC: 0.5128
2025-11-19 10:31:56,366 - INFO - Epoch 960 | Loss: 0.9519 | F1: 0.4200 | Precision: 0.5029 | Recall: 0.5058 | FPR: 0.5298 | AUC: 0.5139
2025-11-19 10:32:02,099 - INFO - Epoch 980 | Loss: 0.9504 | F1: 0.4128 | Precision: 0.5018 | Recall: 0.5035 | FPR: 0.5459 | AUC: 0.5061
2025-11-19 10:32:07,732 - INFO - Epoch 1000 | Loss: 0.9318 | F1: 0.4110 | Precision: 0.5006 | Recall: 0.5011 | FPR: 0.5480 | AUC: 0.5096
2025-11-19 10:32:13,478 - INFO - Epoch 1020 | Loss: 0.9325 | F1: 0.4181 | Precision: 0.5035 | Recall: 0.5071 | FPR: 0.5363 | AUC: 0.5107
2025-11-19 10:32:19,267 - INFO - Epoch 1040 | Loss: 0.9269 | F1: 0.4183 | Precision: 0.5045 | Recall: 0.5091 | FPR: 0.5380 | AUC: 0.5086
2025-11-19 10:32:24,993 - INFO - Epoch 1060 | Loss: 0.9509 | F1: 0.4154 | Precision: 0.5030 | Recall: 0.5060 | FPR: 0.5419 | AUC: 0.5063
2025-11-19 10:32:30,736 - INFO - Epoch 1080 | Loss: 0.9326 | F1: 0.4203 | Precision: 0.5050 | Recall: 0.5099 | FPR: 0.5336 | AUC: 0.5162
2025-11-19 10:32:36,526 - INFO - Epoch 1100 | Loss: 0.9333 | F1: 0.4112 | Precision: 0.5027 | Recall: 0.5055 | FPR: 0.5522 | AUC: 0.5114
2025-11-19 10:32:42,278 - INFO - Epoch 1120 | Loss: 0.9234 | F1: 0.4110 | Precision: 0.5028 | Recall: 0.5056 | FPR: 0.5529 | AUC: 0.5108
2025-11-19 10:32:48,334 - INFO - Epoch 1140 | Loss: 0.9328 | F1: 0.4245 | Precision: 0.5079 | Recall: 0.5159 | FPR: 0.5296 | AUC: 0.5151
2025-11-19 10:32:53,899 - INFO - Epoch 1160 | Loss: 0.9329 | F1: 0.4168 | Precision: 0.5023 | Recall: 0.5046 | FPR: 0.5369 | AUC: 0.5003
2025-11-19 10:32:59,422 - INFO - Epoch 1180 | Loss: 0.9359 | F1: 0.4288 | Precision: 0.5057 | Recall: 0.5115 | FPR: 0.5128 | AUC: 0.5115
2025-11-19 10:33:04,902 - INFO - Epoch 1200 | Loss: 0.9335 | F1: 0.4153 | Precision: 0.5016 | Recall: 0.5032 | FPR: 0.5391 | AUC: 0.5098
2025-11-19 10:33:10,377 - INFO - Epoch 1220 | Loss: 0.9407 | F1: 0.4183 | Precision: 0.5037 | Recall: 0.5075 | FPR: 0.5360 | AUC: 0.5107
2025-11-19 10:33:15,976 - INFO - Epoch 1240 | Loss: 0.9323 | F1: 0.4211 | Precision: 0.5032 | Recall: 0.5064 | FPR: 0.5273 | AUC: 0.5079
2025-11-19 10:33:21,435 - INFO - Epoch 1260 | Loss: 0.9416 | F1: 0.4097 | Precision: 0.5028 | Recall: 0.5057 | FPR: 0.5563 | AUC: 0.5136
2025-11-19 10:33:26,961 - INFO - Epoch 1280 | Loss: 0.9330 | F1: 0.4120 | Precision: 0.5046 | Recall: 0.5092 | FPR: 0.5542 | AUC: 0.5184
2025-11-19 10:33:32,473 - INFO - Epoch 1300 | Loss: 0.9130 | F1: 0.4121 | Precision: 0.5050 | Recall: 0.5100 | FPR: 0.5549 | AUC: 0.5125
2025-11-19 10:33:38,125 - INFO - Epoch 1320 | Loss: 0.9198 | F1: 0.4138 | Precision: 0.5006 | Recall: 0.5012 | FPR: 0.5408 | AUC: 0.5142
2025-11-19 10:33:43,570 - INFO - Epoch 1340 | Loss: 0.9121 | F1: 0.4246 | Precision: 0.5048 | Recall: 0.5097 | FPR: 0.5220 | AUC: 0.5081
2025-11-19 10:33:49,021 - INFO - Epoch 1360 | Loss: 0.9536 | F1: 0.4152 | Precision: 0.5040 | Recall: 0.5081 | FPR: 0.5450 | AUC: 0.5171
2025-11-19 10:33:54,372 - INFO - Epoch 1380 | Loss: 0.9311 | F1: 0.4169 | Precision: 0.5070 | Recall: 0.5139 | FPR: 0.5469 | AUC: 0.5141
2025-11-19 10:33:59,898 - INFO - Epoch 1400 | Loss: 0.9284 | F1: 0.4095 | Precision: 0.5021 | Recall: 0.5041 | FPR: 0.5551 | AUC: 0.5045
2025-11-19 10:34:05,332 - INFO - Epoch 1420 | Loss: 0.9182 | F1: 0.4135 | Precision: 0.5058 | Recall: 0.5115 | FPR: 0.5530 | AUC: 0.5167
2025-11-19 10:34:10,795 - INFO - Epoch 1440 | Loss: 0.9227 | F1: 0.4221 | Precision: 0.5054 | Recall: 0.5109 | FPR: 0.5299 | AUC: 0.5134
2025-11-19 10:34:16,531 - INFO - Epoch 1460 | Loss: 0.9262 | F1: 0.4166 | Precision: 0.5048 | Recall: 0.5096 | FPR: 0.5430 | AUC: 0.5117
2025-11-19 10:34:21,963 - INFO - Epoch 1480 | Loss: 0.9096 | F1: 0.4221 | Precision: 0.5032 | Recall: 0.5065 | FPR: 0.5247 | AUC: 0.5079
2025-11-19 10:34:27,525 - INFO - Epoch 1500 | Loss: 0.9069 | F1: 0.4239 | Precision: 0.5062 | Recall: 0.5124 | FPR: 0.5270 | AUC: 0.5148
2025-11-19 10:34:32,988 - INFO - Epoch 1520 | Loss: 0.9187 | F1: 0.4249 | Precision: 0.5058 | Recall: 0.5116 | FPR: 0.5235 | AUC: 0.5109
2025-11-19 10:34:38,821 - INFO - Epoch 1540 | Loss: 0.9321 | F1: 0.4126 | Precision: 0.5060 | Recall: 0.5119 | FPR: 0.5557 | AUC: 0.5167
2025-11-19 10:34:44,408 - INFO - Epoch 1560 | Loss: 0.9247 | F1: 0.4191 | Precision: 0.5039 | Recall: 0.5079 | FPR: 0.5344 | AUC: 0.5111
2025-11-19 10:34:50,150 - INFO - Epoch 1580 | Loss: 0.9109 | F1: 0.4156 | Precision: 0.5032 | Recall: 0.5065 | FPR: 0.5422 | AUC: 0.5113
2025-11-19 10:34:55,609 - INFO - Epoch 1600 | Loss: 0.9098 | F1: 0.4123 | Precision: 0.5024 | Recall: 0.5047 | FPR: 0.5487 | AUC: 0.5112
2025-11-19 10:35:01,603 - INFO - Epoch 1620 | Loss: 0.9259 | F1: 0.4154 | Precision: 0.5047 | Recall: 0.5094 | FPR: 0.5458 | AUC: 0.5171
2025-11-19 10:35:07,195 - INFO - Epoch 1640 | Loss: 0.9108 | F1: 0.4205 | Precision: 0.5051 | Recall: 0.5102 | FPR: 0.5337 | AUC: 0.5153
2025-11-19 10:35:12,855 - INFO - Epoch 1660 | Loss: 0.9166 | F1: 0.4217 | Precision: 0.5037 | Recall: 0.5074 | FPR: 0.5270 | AUC: 0.5084
2025-11-19 10:35:18,325 - INFO - Epoch 1680 | Loss: 0.9155 | F1: 0.4258 | Precision: 0.5084 | Recall: 0.5168 | FPR: 0.5271 | AUC: 0.5169
2025-11-19 10:35:23,864 - INFO - Epoch 1700 | Loss: 0.9191 | F1: 0.4219 | Precision: 0.5034 | Recall: 0.5068 | FPR: 0.5258 | AUC: 0.5096
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

# 子图1：Loss变化
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

# 子图3：Precision变化
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
plt.savefig('GAE_training.png')  # 保存为图片
plt.close()