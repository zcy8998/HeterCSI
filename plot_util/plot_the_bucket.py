import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --------------------------
# 1. IEEE Trans 风格全局设置
# --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,              # IEEE标准通常较小，10-12皆可
    'axes.labelsize': 12,         # 轴标签
    'axes.titlesize': 12,         # 标题
    'legend.fontsize': 10,        # 图例
    'xtick.labelsize': 10,        # X轴刻度
    'ytick.labelsize': 10,        # Y轴刻度
    'axes.linewidth': 1.0,        # 轴线宽度
    'grid.color': '#b0b0b0',      # 稍微深一点的灰色，打印更清晰
    'grid.linestyle': ':',        # 网格线为点线
    'grid.alpha': 0.6,
    'grid.linewidth': 0.8,
    'xtick.direction': 'in',      # 刻度向内
    'ytick.direction': 'in',      # 刻度向内
    'xtick.top': True,            # 上方显示刻度
    'ytick.right': True,          # 右侧显示刻度
    'figure.dpi': 300,            # 高分辨率
    'savefig.bbox': 'tight',      # 去除白边
    'lines.linewidth': 1.5,       # 线宽
    'lines.markersize': 6         # 标记点大小适中
})

# --------------------------
# 2. 数据录入 (根据提供的 Overall Average NMSE)
# --------------------------
# X轴: Bucket sizes
buckets = [1, 2, 4, 8, 16]

# Data: High Heterogeneity (Overall Average NMSE)
# Values: -5.15..., -7.40..., -9.15..., -8.55..., -8.46...
high_hetero_data = [
    -5.1575952, 
    -7.4036522, 
    -9.1546164, 
    -8.5532656, 
    -8.4622364
]

# Data: Low Heterogeneity (Overall Average NMSE)
# Values: -2.42..., -2.72..., -2.59..., -2.59..., -2.44...
low_hetero_data  = [
    -2.4285100, 
    -2.7215242, 
    -2.5983577, 
    -2.5961118, 
    -2.4479625
]

# --------------------------
# 3. 绘图逻辑
# --------------------------
# 尺寸建议：IEEE 单栏宽度约为 3.5英寸。
# 设置为 (5, 3.8) 既能保证清晰度，插入文档时缩小也不会模糊。
fig, ax = plt.subplots(figsize=(5, 3.8)) 

# 绘制 High Heterogeneity (蓝色, 圆点, 实线)
ax.plot(buckets, high_hetero_data, 
        marker='o',              
        linestyle='-',           
        color='#0072BD',         # IEEE 常用深蓝
        label='$\mathcal{H}_{CSI} = 57.20\%$', 
        markeredgecolor='white', # 标记边缘白色，增加对比度
        markeredgewidth=0.8,
        zorder=3)                # 保证线在网格之上

# 绘制 Low Heterogeneity (红色, 方块, 虚线)
ax.plot(buckets, low_hetero_data, 
        marker='s',              
        linestyle='--',          
        color='#D95319',         # IEEE 常用深红/橙红
        label='$\mathcal{H}_{CSI} = 0\%$', 
        markeredgecolor='white',
        markeredgewidth=0.8,
        zorder=3)

# --------------------------
# 4. 坐标轴与美化
# --------------------------
ax.set_xlabel('Number of Buckets')
ax.set_ylabel('NMSE (dB)')

# X轴设置: Log2 刻度
ax.set_xscale('log', base=2)
ax.set_xticks(buckets)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()) 

# Y轴设置:
# 数据范围约在 -10 到 -2 之间。设置一定的 margin 让图不顶格。
# 为了美观，可以强制反转坐标轴（如果想表示越低越好，通常保持原样即可，NMSE通常是越低越好）
# 这里保持数学上的直观性（负数在下）。
ax.set_ylim(-10.5, -1.0) 
ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) # 每隔2dB一个刻度

# 网格
ax.grid(True, which='major', axis='both')

# 图例
# frameon=True 加上边框，framealpha=1 不透明，避免遮挡网格混乱
ax.legend(loc='lower left', frameon=True, edgecolor='black', framealpha=1.0, fancybox=False)

# --------------------------
# 5. 保存
# --------------------------
plt.tight_layout()
# 建议同时保存为 PDF (矢量图，适合插入 LaTeX) 和 PNG (预览)
plt.savefig('bucket_nmse_comparison.pdf', format='pdf') 
plt.savefig('bucket_nmse_comparison.png', dpi=300)
plt.show()