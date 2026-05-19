import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc


model_RGCN_XGBoost_fpr = [0.0, 0.0270, 0.0433, 0.0551, 0.0695, 0.0959, 1.0]  # 补充0和1，按FPR升序
model_RGCN_XGBoost_tpr = [0.0, 0.6854, 0.7581, 0.7963, 0.8300, 0.8659, 1.0]
model_RGCN_XGBoost_auc = 0.9357

model_RGCN_RF_fpr = [0.0,0.0094, 0.0174, 0.0244, 0.0312, 0.0471, 1.0]
model_RGCN_RF_tpr = [0.0, 0.4337, 0.5154, 0.5648, 0.6060, 0.6839, 1.0]
model_RGCN_RF_auc = 0.9248

model_RGCN_fpr = [0.0,0.0456,0.0886,0.1264,0.1758,0.2748, 1.0]
model_RGCN_tpr = [0.0, 0.4289, 0.5704, 0.6413, 0.7159,  0.8098, 1.0]
model_RGCN_auc = 0.8487


plt.figure(figsize=(8, 6))

plt.plot(model_RGCN_XGBoost_fpr, model_RGCN_XGBoost_tpr, color='darkorange', lw=2,
         label=f'RGCN-XGBoost (AUC = {model_RGCN_XGBoost_auc:.4f})')

plt.plot(model_RGCN_RF_fpr, model_RGCN_RF_tpr, color='blue', lw=2, linestyle='--',
         label=f'RGCN-RF (AUC = {model_RGCN_RF_auc:.4f})')

plt.plot(model_RGCN_fpr, model_RGCN_tpr, color='red', lw=2, linestyle='--',
         label=f'RGCN (AUC = {model_RGCN_auc:.4f})')
# 绘制随机猜测基线（AUC=0.5）
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='AUC=0.5')

# ===================== 3. 图表美化与标注 =====================
plt.xlim([0.0, 1.0])  # 横轴范围（FPR）
plt.ylim([0.0, 1.05]) # 纵轴范围（TPR）
plt.xlabel('FPR', fontsize=12)
plt.ylabel('TPR', fontsize=12)
plt.title('ROC', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)  # 网格线

# 保存图片（可选）
plt.savefig('model_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()