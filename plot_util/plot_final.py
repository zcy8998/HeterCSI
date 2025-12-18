import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置科研风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 16,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'lightgrey',
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})

# 示例数据（请替换为您的实际数据）
num_datasets = np.array([16, 24, 32])  # 数据集数量

# 四种方案的NMSE数据
# global_nmse = np.array([0.258, 0.166, 0.145])       # Global方案
# proposed_nmse = np.array([0.22, 0.164, 0.143])     # Proposed方案
# bucket_nmse = np.array([0.212, 0.166, 0.158])       # Bucket方案
# sequential_nmse = np.array([0.462, 0.483, 0.463])   # Sequential方案

global_nmse = np.array([0.258, 0.166, 0.145])       # Global方案
proposed_nmse = np.array([0.22, 0.164, 0.143])     # Proposed方案
bucket_nmse = np.array([0.462, 0.483, 0.457])       # Bucket方案
# sequential_nmse = np.array([0.462, 0.483, 0.463])   # Sequential方案

# 创建图表
fig, ax = plt.subplots(figsize=(7, 6))

# 绘制四种方案的折线图
ax.plot(num_datasets, global_nmse, 
        'o-', color='#1f77b4', linewidth=2, markersize=8, label='Global')
ax.plot(num_datasets, proposed_nmse, 
        's--', color='#d62728', linewidth=2, markersize=8, label='Proposed')
ax.plot(num_datasets, bucket_nmse, 
        'D-.', color='#2ca02c', linewidth=2, markersize=8, label='Sequential')
# ax.plot(num_datasets, sequential_nmse, 
#         '^:', color='#9467bd', linewidth=2, markersize=8, label='Sequential')

# 设置坐标轴
ax.set_xscale('log', base=2)
ax.set_xticks(num_datasets)
ax.set_xticklabels([f'{x}' for x in num_datasets])
ax.set_xlabel('Number of Datasets', fontweight='bold')
ax.set_ylabel('NMSE', fontweight='bold')
# ax.set_ylim(bottom=0)  # 确保y轴从0开始

# 添加网格和图例
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(frameon=True, shadow=True, loc='best')

# 优化布局并保存
plt.tight_layout()
plt.savefig('results/final1.pdf', bbox_inches='tight')
# plt.savefig('results/nmse_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()