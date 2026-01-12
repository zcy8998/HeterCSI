import matplotlib.pyplot as plt
import numpy as np

# --- 1. 设置 IEEE Transactions 科研绘图风格 ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',       # 数学公式字体类似 Times
    'font.size': 14,                  # 全局字号
    'axes.labelsize': 14,             # 轴标签字号
    'legend.fontsize': 14,            # 图例字号
    'xtick.labelsize': 14,            # X轴刻度字号
    'ytick.labelsize': 14,            # Y轴刻度字号
    'axes.linewidth': 1.0,            # 边框粗细
    'axes.edgecolor': 'black',        # 边框颜色
    'xtick.direction': 'in',          # 刻度朝内 (IEEE风格)
    'ytick.direction': 'in',          # 刻度朝内
    'lines.linewidth': 1.5,           # 线条粗细
    'lines.markersize': 7,            # 标记点大小
    'grid.linestyle': '--',           # 网格虚线
    'grid.alpha': 0.4                 # 网格透明度
})

# --- 2. 录入数据 (来自表格截图) ---
# X轴：Number of Datasets
num_datasets = np.array([8, 16, 24, 32, 40])

# Y轴：Training Time (minutes)
# Global
data_global = np.array([2.54, 5.08, 6.66, 13.11, 30.55])

# Proposed (B=4)
data_prop_b4 = np.array([2.05, 4.12, 5.54, 8.60, 14.41])

# Proposed (B=8)
data_prop_b8 = np.array([2.02, 4.09, 5.15, 7.37, 11.61])

# Alternating
data_alternating = np.array([2.03, 4.11, 4.88, 6.74, 9.60])

# --- 3. 定义样式字典 (参考您的要求) ---
# 注意：Proposed (B=4) 使用了与 Proposed 相同的色系，但改变 LineStyle 以示区分
styles = {
    'Global': {
        'marker': 'o', 'color': '#0072BD', 'ls': '--', 'label': 'Global'
    },
    'Proposed (B=4)': {
        'marker': 'd', 'color': '#D95319', 'ls': '--', 'label': 'Proposed (B=4)'
    },
    'Proposed (B=8)': {
        'marker': 's', 'color': '#D95319', 'ls': '-',  'label': 'Proposed (B=8)'
    },
    'Alternating': {
        'marker': '^', 'color': '#77AC30', 'ls': '-.', 'label': 'Alternating'
    }
}

# --- 4. 创建图表 ---
fig, ax = plt.subplots(figsize=(6, 5))

# 绘制线条
ax.plot(num_datasets, data_global, **styles['Global'])
ax.plot(num_datasets, data_prop_b4, **styles['Proposed (B=4)'])
ax.plot(num_datasets, data_prop_b8, **styles['Proposed (B=8)'])
ax.plot(num_datasets, data_alternating, **styles['Alternating'])

# --- 5. 坐标轴设置 ---
ax.set_xlabel('Number of Datasets')
ax.set_ylabel('Training Time (min)')

# 设置X轴刻度严格对应数据点
ax.set_xticks(num_datasets)
ax.set_xticklabels(num_datasets)

# 开启网格
ax.grid(True)

# --- 6. 图例设置 ---
# frameon=True 显示图例边框 (IEEE通常有边框)，loc='best'自动寻找最佳位置
ax.legend(frameon=True, edgecolor='black', fancybox=False, loc='upper left')

# --- 7. 保存与展示 ---
plt.tight_layout()

# 保存为矢量图 PDF (推荐用于论文) 和高也就是 PNG
plt.savefig('results/training_time_comparison.pdf', bbox_inches='tight')
plt.savefig('results/training_time_comparison.png', dpi=300, bbox_inches='tight')

plt.show()