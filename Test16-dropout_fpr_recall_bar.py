import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 示例数据（可替换为你的实验数据）
dropout = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # dropout值
fpr_values = [0.0537, 0.0522, 0.0522, 0.0491, 0.0466, 0.0530]  # FPR
recall_values = [0.8608, 0.8608,0.8597, 0.8616, 0.8606, 0.8608]  # Recall

# 绘制柱状图
x = np.arange(len(dropout))  # x轴位置
width = 0.35  # 柱状图宽度

fig, ax = plt.subplots(figsize=(10, 8))  # 增加图表高度

# 绘制FPR和Recall的柱状图
rects1 = ax.bar(x - width/2, fpr_values, width, label='FPR', color='#FF4500', alpha=0.7)
rects2 = ax.bar(x + width/2, recall_values, width, label='Recall', color='#008B8B', alpha=0.7)

# 添加数据标签
def add_labels(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 增加垂直偏移量
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

add_labels(rects1, ax)
add_labels(rects2, ax)

# 图表样式设置
ax.set_title('Dropout与 FPR/Recall 关系', fontsize=14, fontweight='bold')
ax.set_xlabel('Dropout值', fontsize=14)
ax.set_ylabel('指标值', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(dropout)

# 调整y轴范围，增加顶部空间
ax.set_ylim(0, max(max(fpr_values), max(recall_values)) * 1.1)  # 顶部留出10%空间

ax.grid(True, linestyle='--', alpha=0.7, axis='y')  # 添加水平网格线
ax.legend(loc='upper right')  # 显示图例

plt.tight_layout()  # 自动调整布局

# 保存图表
plt.savefig('dropout_fpr_recall_bar.png', dpi=300, bbox_inches='tight')
print("优化后的柱状图已保存为 dropout_fpr_recall_bar.png")