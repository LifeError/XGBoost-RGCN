import matplotlib.pyplot as plt
import numpy as np

# 设置支持中文的字体（仅保留黑体）
plt.rcParams['font.family'] = 'SimHei'  # 黑体是Windows/macOS/Linux常见预装字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 评估指标数据
metrics = ['F1', 'Precision', 'Recall', '1-FPR', 'AUC']
values = [0.8560, 0.8519, 0.8604, 0.0448, 0.9356]

# 由于FPR是越小越好，将其转换为越大越好的指标 (1-FPR)
values[metrics.index('1-FPR')] = 1 - values[metrics.index('1-FPR')]

# 创建雷达图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)

# 计算每个指标的角度
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()

# 闭合雷达图
values = values + values[:1]
angles = angles + angles[:1]

# 绘制雷达图
ax.plot(angles, values, 'o-', linewidth=2, color='#4a8fe7')
ax.fill(angles, values, alpha=0.25, color='#4a8fe7')

# 设置坐标轴标签和标题
ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=12)
ax.set_ylim(0, 1)
plt.title('最终模型性能雷达图', fontsize=16, fontweight='bold', pad=20)

# 添加数据标签
for i, value in enumerate(values[:-1]):
    angle_rad = angles[i]
    if angle_rad > np.pi/2 and angle_rad < 3*np.pi/2:  # 左侧标签
        ha = 'right'
    else:  # 右侧标签
        ha = 'left'
    ax.annotate(f'{value:.4f}',
                xy=(angle_rad, value),
                xytext=(angle_rad, value + 0.05),
                ha=ha,
                fontsize=10)

# 美化图表
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存图表
plt.savefig('model_metrics_radar.png', dpi=300, bbox_inches='tight')
print("模型评估指标雷达图已保存为 model_metrics_radar.png")