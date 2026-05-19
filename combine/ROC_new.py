import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc


# ========== 1. 补充完整的模型数据（确保FPR是升序） ==========
# RGCN-XGBoost（完整数据）
model_RGCN_XGBoost_fpr = [0.0,0.0270, 0.0350, 0.0430, 0.0484, 0.0546, 0.0617, 0.0692, 0.0798, 0.0957, 1.0]
model_RGCN_XGBoost_tpr = [0.0, 0.6846, 0.7318, 0.7581, 0.7813, 0.7948, 0.8150, 0.8307, 0.8427, 0.8637, 1.0]
model_RGCN_XGBoost_auc = 0.9357

# RGCN-RF（补充完整数据，避免点过少）
model_RGCN_RF_fpr = [0.0,0.0093, 0.0134, 0.0164, 0.0195, 0.0233, 0.0261, 0.0300, 0.0363, 0.0447, 1.0]
model_RGCN_RF_tpr = [0.0, 0.4202, 0.4704, 0.5064, 0.5333, 0.5521, 0.5738, 0.5993, 0.6300, 0.6719, 1.0]
model_RGCN_RF_auc = 0.9260

# RGCN（完整数据）
model_RGCN_fpr = [0.0,0.0885, 0.1245, 0.1505, 0.1768, 0.2050, 0.2336, 0.2672, 0.3110, 0.3850, 1.0]
model_RGCN_tpr = [0.0, 0.5784, 0.6483, 0.7002, 0.7364, 0.7646, 0.7891, 0.8148, 0.8427, 0.8772, 1.0]
model_RGCN_auc = 0.8523


# ========== 2. 生成平滑的FPR序列+插值TPR（消除折点） ==========
# 生成100个均匀分布的FPR点，让曲线更连续
smooth_fpr = np.linspace(0, 1, 100)

# 对每个模型的TPR进行插值（匹配平滑FPR）
xgb_tpr_smooth = np.interp(smooth_fpr, model_RGCN_XGBoost_fpr, model_RGCN_XGBoost_tpr)
rf_tpr_smooth = np.interp(smooth_fpr, model_RGCN_RF_fpr, model_RGCN_RF_tpr)
rgcn_tpr_smooth = np.interp(smooth_fpr, model_RGCN_fpr, model_RGCN_tpr)


# ========== 3. 绘制平滑的ROC曲线 ==========
plt.figure(figsize=(8, 6))

# 绘制平滑后的曲线
plt.plot(smooth_fpr, xgb_tpr_smooth, color='darkorange', lw=2,
         label=f'RGCN-XGBoost (AUC = {model_RGCN_XGBoost_auc:.4f})')

plt.plot(smooth_fpr, rf_tpr_smooth, color='blue', lw=2,
         label=f'RGCN-RF (AUC = {model_RGCN_RF_auc:.4f})')

plt.plot(smooth_fpr, rgcn_tpr_smooth, color='red', lw=2,
         label=f'RGCN (AUC = {model_RGCN_auc:.4f})')

# 随机猜测基线
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='AUC=0.5')


# 图表美化
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR', fontsize=12)
plt.ylabel('TPR', fontsize=12)
plt.title('ROC curve', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

# 保存+显示
plt.savefig('model_roc_comparison_smooth.png', dpi=300, bbox_inches='tight')
plt.show()