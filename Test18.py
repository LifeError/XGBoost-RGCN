import os
from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image

# 指定Graphviz路径
os.environ["PATH"] += os.pathsep + r'E:\Anaconda\Graphviz\bin'

# 设置Matplotlib后端为Agg（非交互式）
plt.switch_backend('Agg')

# 创建有向图
dot = Digraph(comment='RGCN Model Computation',
              node_attr={'fontname': 'Helvetica,Arial,sans-serif'},
              edge_attr={'fontname': 'Helvetica,Arial,sans-serif'})
dot.attr(rankdir='TB', size='8,8', dpi='300')

# 添加节点
dot.node('input', '输入特征 (x)')
dot.node('edge_index', '边索引 (edge_index)')
dot.node('edge_type', '边类型 (edge_type)')
dot.node('input_layer', '输入层 (RGCNConv)[34,128]')
dot.node('hidden_layer_1', '隐藏层1 (RGCNConv)[128,128]')
dot.node('dropout_1', 'Dropout层1')
dot.node('hidden_layer_2', '隐藏层2 (RGCNConv)[128,128]')
dot.node('dropout_2', 'Dropout层2')
dot.node('output_layer', '输出层 (Linear)[128,2]')
dot.node('output', '输出结果')

# 添加边
dot.edge('input', 'input_layer', label='特征输入')
dot.edge('edge_index', 'input_layer', label='边索引输入')
dot.edge('edge_type', 'input_layer', label='边类型输入')
dot.edge('input_layer', 'hidden_layer_1', label='ReLU激活')
dot.edge('hidden_layer_1', 'dropout_1', label='特征传递')
# 修改此处，让Dropout1直接连接到隐藏层2
dot.edge('dropout_1', 'hidden_layer_2', label='Dropout后输出')
dot.edge('hidden_layer_2', 'dropout_2', label='特征传递')
dot.edge('dropout_2', 'output_layer', label='Dropout后输出')
dot.edge('output_layer', 'output', label='最终输出')

# 渲染图形
img_path = 'rgcn_model_computation'
dot.render(img_path, format='png', cleanup=True, view=False)

# 使用matplotlib保存图像（不显示）
img = Image.open(f'{img_path}.png')
plt.rcParams['figure.dpi'] = 300
plt.imshow(img)
plt.axis('off')
plt.savefig(f'{img_path}_matplotlib.png', bbox_inches='tight')

print(f"图像已保存为: {img_path}_matplotlib.png")