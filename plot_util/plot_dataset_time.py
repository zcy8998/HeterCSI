import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# --- 1. 设置科研风格 (保持您提供的样式不变) ---
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

# --- 2. 录入数据 (来自您的表格截图) ---
# 横轴：数据集数量
num_datasets = np.array([8, 16, 24, 32])

# 纵轴：每个epoch消耗的时间 (s)
# 对应表格行: "all"
time_all = np.array([264, 520, 927, 1567])

# 对应表格行: "group(proposed)"
time_proposed = np.array([190, 382, 812, 1125])

# 对应表格行: "seq_new"
time_seq_new = np.array([188, 380, 810, 1120])

# --- 3. 创建图表 ---
fig, ax = plt.subplots(figsize=(7, 6))

# --- 4. 绘制折线图 ---
# 方案1: all
ax.plot(num_datasets, time_all, 
        'o-', color='#1f77b4', linewidth=2, markersize=8, label='Global')

# 方案2: group(proposed)
ax.plot(num_datasets, time_proposed, 
        's--', color='#d62728', linewidth=2, markersize=8, label='Proposed')

# 方案3: seq_new
ax.plot(num_datasets, time_seq_new, 
        'D-.', color='#2ca02c', linewidth=2, markersize=8, label='Alternating')


# --- 5. 设置坐标轴 ---
# 注意：您的数据 8, 16, 24, 32 是线性增长的（每步+8），通常使用线性坐标轴视觉效果更好。
# 如果必须使用 Log 轴（如参考代码），请取消下面这行的注释：
# ax.set_xscale('log', base=2) 

ax.set_xticks(num_datasets)
ax.set_xticklabels([f'{x}' for x in num_datasets])

ax.set_xlabel('Number of Datasets', fontweight='bold')
ax.set_ylabel('Time per Epoch (s)', fontweight='bold')

# --- 6. 添加网格和图例 ---
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(frameon=True, shadow=True, loc='best')

# --- 7. 优化布局并保存 ---
plt.tight_layout()

# 保存为PDF或PNG
plt.savefig('results/time_consumption_comparison.pdf', bbox_inches='tight')
plt.savefig('results/time_consumption_comparison.png', dpi=300, bbox_inches='tight')

# 显示图表 (如果在Jupyter/IDE中运行)
plt.show()
