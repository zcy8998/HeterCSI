import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --------------------------
# 1. IEEE Trans 风格全局设置
# --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,              # 基础字号
    'axes.labelsize': 14,         # 轴标签字号
    'axes.titlesize': 14,         # 标题字号
    'legend.fontsize': 12,        # 图例字号
    'xtick.labelsize': 12,        # X轴刻度字号
    'ytick.labelsize': 12,        # Y轴刻度字号
    'axes.linewidth': 1.0,        # 轴线宽度
    'grid.color': 'gray',
    'grid.linestyle': ':',        # 网格线改为点线
    'grid.alpha': 0.5,
    'grid.linewidth': 0.5,
    'xtick.direction': 'in',      # 刻度向内
    'ytick.direction': 'in',      # 刻度向内
    'xtick.top': True,            # 上方显示刻度
    'ytick.right': True,          # 右侧显示刻度
    'figure.dpi': 300,            # 高分辨率
    'savefig.bbox': 'tight',      # 保存时去除白边
    'lines.linewidth': 1.5,       # 线宽
    'lines.markersize': 8         # 标记点大小
})

# --------------------------
# 2. 数据录入
# --------------------------
# X轴: Bucket sizes
buckets = [1, 2, 4, 8, 16]

# Y轴: Average values (High)
high_data = [0.233, 0.154, 0.105, 0.123, 0.138]

# Y轴: Average values (Low)
low_data  = [0.701, 0.646, 0.649, 0.649, 0.676]

# --------------------------
# 3. 绘图逻辑
# --------------------------
fig, ax = plt.subplots(figsize=(7, 5)) # 典型的单栏/半页图尺寸

# 绘制 High (蓝色, 圆点实线)
ax.plot(buckets, high_data, 
        marker='o',              # 圆点
        linestyle='-',           # 实线
        color='#1f77b4',         # 经典蓝
        label='High', 
        markeredgecolor='white', # 标记边缘白色，增加清晰度
        markeredgewidth=1.0)

# 绘制 Low (红色, 方块虚线) - 使用虚线以区分不同类别
ax.plot(buckets, low_data, 
        marker='s',              # 方块
        linestyle='--',          # 虚线
        color='#d62728',         # 经典红
        label='Low', 
        markeredgecolor='white',
        markeredgewidth=1.0)

# --------------------------
# 4. 坐标轴与美化
# --------------------------
ax.set_xlabel('Bucket Size')
ax.set_ylabel('Average Metric')  # 请根据实际含义修改，例如 "NMSE" 或 "Error Rate"

# 关键设置：X轴使用 Log2 刻度
# 因为 bucket 是 1, 2, 4, 8, 16 (2的幂次)，线性坐标会挤在一起
# Log坐标能让它们均匀分布
ax.set_xscale('log', base=2)
ax.set_xticks(buckets)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()) # 显示 "1", "2"... 而不是 "2^0"

# Y轴范围设置 (根据数据范围留出一点余量)
ax.set_ylim(0, 0.8) 

# 网格
ax.grid(True, which='major', axis='both')

# 图例
ax.legend(loc='best', frameon=True, edgecolor='black', framealpha=1.0, fancybox=False)

# --------------------------
# 5. 保存
# --------------------------
plt.tight_layout()
plt.savefig('results/bucket_line_plot.png', dpi=300)
plt.savefig('results/bucket_line_plot.pdf', format='pdf') # 推荐论文使用PDF
plt.show()