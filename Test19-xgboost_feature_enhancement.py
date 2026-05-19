import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.font_manager as fm


# 设置中文字体
def set_chinese_font():
    chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC',
                     'Microsoft YaHei', 'Arial Unicode MS']
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            print(f"已设置中文字体: {font}")
            return True

    print("警告: 未找到可用的中文字体，将使用默认字体")
    return False


set_chinese_font()
plt.rcParams["axes.unicode_minus"] = False  # 确保负号正确显示

# 创建画布
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')  # 隐藏坐标轴

# 定义颜色
color_text = 'black'
color_arrow = '#333333'


# 添加文本框函数
def add_text_box(x, y, text, width=1.8, height=1.2, fontsize=12, color=color_text,
                 border_color='lightgray', bg_color='white', ha='center', va='center'):
    box = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=2,
                            edgecolor=border_color, facecolor=bg_color, alpha=1.0,
                            linestyle='-', capstyle='round', joinstyle='round')
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize, ha=ha, va=va, color=color)
    return {
        'x': x, 'y': y, 'width': width, 'height': height,
        'left': x - width / 2, 'right': x + width / 2,
        'top': y + height / 2, 'bottom': y - height / 2
    }


# 添加箭头函数 - 精确控制箭头起止位置
def add_arrow(box1, box2, direction, width=0.015, head_width=0.25, head_length=0.3, color=color_arrow):
    if direction == 'right':
        x1 = box1['right']  # 起点：第一个框的右侧
        y1 = box1['y']
        x2 = box2['left']  # 终点：第二个框的左侧
        y2 = box2['y']
    elif direction == 'left':
        x1 = box1['left']
        y1 = box1['y']
        x2 = box2['right']
        y2 = box2['y']
    elif direction == 'down':
        x1 = box1['x']
        y1 = box1['bottom']
        x2 = box2['x']
        y2 = box2['top']
    elif direction == 'up':
        x1 = box1['x']
        y1 = box1['top']
        x2 = box2['x']
        y2 = box2['bottom']
    elif direction == 'diag_down':
        x1 = box1['right']
        y1 = box1['y']
        x2 = box2['left']
        y2 = box2['top']
    elif direction == 'diag_up':
        x1 = box1['right']
        y1 = box1['y'] - 0.2  # 微调以避免重叠
        x2 = box2['left']
        y2 = box2['bottom']

    # 添加小的偏移，使箭头不直接接触框
    offset = 0.05

    if direction in ['right', 'left', 'diag_down', 'diag_up']:
        x1 += offset if direction in ['right', 'diag_down', 'diag_up'] else -offset
        x2 += offset if direction == 'left' else -offset
    if direction in ['down', 'up', 'diag_down', 'diag_up']:
        y1 += offset if direction == 'up' else -offset
        y2 += offset if direction == 'down' else -offset

    ax.arrow(x1, y1, x2 - x1, y2 - y1, width=width, head_width=head_width, head_length=head_length,
             fc=color, ec=color, length_includes_head=True)


# 添加标题
plt.title('XGBoost特征增强流程图', fontsize=16)

# 统一框的宽度和高度
box_width = 1.8
box_height = 1.2

# 1. 原始数据
box1 = add_text_box(2, 6.5, "原始特征数据\n(45954, 32)",
                    width=box_width, height=box_height, border_color=color_text)

# 2. 数据划分
box2 = add_text_box(2, 4.5, "数据划分\n训练集/验证集/测试集",
                    width=box_width, height=box_height, border_color=color_text)
add_arrow(box1, box2, 'down')  # 从数据到划分的箭头

# 3. XGBoost模型
box3 = add_text_box(2, 2.5, "XGBoost模型\n使用100课决策树",
                    width=box_width, height=box_height, border_color=color_text)
add_arrow(box2, box3, 'down')  # 从划分到模型的箭头

# 4. 概率预测
box4 = add_text_box(5, 2.5, "输出概率预测\n(45954, 2)",
                    width=box_width, height=box_height, border_color=color_text)
add_arrow(box3, box4, 'right')  # 从模型到预测的箭头

# 5. 特征拼接
box5 = add_text_box(8, 2.5, "与原始特征进行拼接",
                    width=box_width, height=box_height, border_color=color_text)
add_arrow(box4, box5, 'right')  # 从预测到拼接的箭头

# 6. 增强特征
box6 = add_text_box(8, 4.5, "增强特征数据\n(45954, 34)",
                    width=box_width, height=box_height, border_color=color_text)
add_arrow(box5, box6, 'up')  # 从拼接数据到增强数据的箭头

# 7. RGCN模型
box7 = add_text_box(8, 6.5, "特征归一化\n送入RGCN模型",
                    width=box_width, height=box_height, border_color=color_text)
add_arrow(box6, box7, 'up')  # 从增强数据到RGCN的箭头

# 8. 从原始数据直接到拼接的箭头
add_arrow(box1, box5, 'diag_down')



# 保存图形
plt.tight_layout()
plt.savefig('xgboost_feature_enhancement.png', dpi=300, bbox_inches='tight')
print("已生成特征增强流程图")
