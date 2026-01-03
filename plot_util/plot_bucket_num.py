import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

# --------------------------
# 1. IEEE 风格全局设置
# --------------------------
# 使用 Times New Roman 字体，符合 IEEE 论文标准
config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",  # 数学公式字体更接近 LaTeX
    "font.size": 12,             # 基础字号
    "axes.labelsize": 14,        # 轴标签字号
    "legend.fontsize": 12,       # 图例字号
    "xtick.labelsize": 12,       # X轴刻度字号
    "ytick.labelsize": 12,       # Y轴刻度字号
    "axes.linewidth": 1.0,       # 边框粗细
    "grid.linestyle": "--",      # 网格虚线
    "grid.alpha": 0.5,           # 网格透明度
    "figure.dpi": 300,           #以此分辨率保存
}
plt.rcParams.update(config)

# --------------------------
# 2. 数据准备 (更新为表格数据)
# --------------------------
bucket_sizes = [1, 2, 4, 8, 16]

# 更新为图片中的表格数据
nmse_random =   [0.424979,  0.1790774, 0.0477455, 0.1024281, 0.1274655]
nmse_temporal = [0.188075,  0.1832492, 0.1855076, 0.1850721, 0.2043378]
nmse_freq =     [0.0870814, 0.1008942, 0.0811304, 0.0803556, 0.0831827]

# 计算平均值
data_matrix = np.array([nmse_random, nmse_temporal, nmse_freq])
nmse_average = np.mean(data_matrix, axis=0)

# --------------------------
# 3. 绘图逻辑
# --------------------------
fig, ax = plt.subplots(figsize=(6, 6)) # 典型的单栏/半页宽度

# --- A. 绘制各个方法的线 (背景参考) ---
# IEEE风格通常用不同的 LineStyle 和 Marker 来区分，而不仅仅是颜色
# Random: 蓝色圆圈
ax.plot(bucket_sizes, nmse_random, 
        marker='o', markersize=6, linestyle='-.', linewidth=1.2,
        color='#1f77b4', alpha=0.6, label='Random-0.85')

# Temporal: 红色方块
ax.plot(bucket_sizes, nmse_temporal, 
        marker='s', markersize=6, linestyle='--', linewidth=1.2,
        color='#d62728', alpha=0.6, label='Temporal-0.5')

# Freq: 绿色菱形
ax.plot(bucket_sizes, nmse_freq, 
        marker='D', markersize=6, linestyle=':', linewidth=1.2,
        color='#2ca02c', alpha=0.6, label='Freq-0.5')

# --- B. 绘制平均趋势线 (主角) ---
# 颜色：使用深紫色 (#6A0DAD) 代替黑色，既显眼又不单调，打印清晰
# 样式：实线，加粗，放在最上层 (zorder=10)
ax.plot(bucket_sizes, nmse_average, 
        marker='*', markersize=12, linestyle='-', linewidth=2.5,
        color='#6A0DAD', label='Average Trend', zorder=10)

# --------------------------
# 4. 坐标轴与细节调整
# --------------------------
ax.set_xlabel('Number of Buckets')
ax.set_ylabel('NMSE')  # 简化Y轴标签，或写 'Average NMSE'

# --- 关键修改：使用对数坐标轴 ---
# 因为 bucket 是 1, 2, 4, 8, 16 倍增的，对数轴能让间距相等，更符合逻辑
ax.set_xscale('log', base=2)
ax.set_xticks(bucket_sizes)
ax.get_xaxis().set_major_formatter(ScalarFormatter()) # 强制显示为 1, 2, 4 而不是 2^0, 2^1

# 开启网格
ax.grid(True, which='major', linestyle='--', alpha=0.5)

# 图例设置 (IEEE 风格通常要求图例简洁)
handles, labels = ax.get_legend_handles_labels()
# 将 Average 放在第一个位置
order = [3, 0, 1, 2] 
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
          loc='best', edgecolor='black', fancybox=False, framealpha=1.0)

# --------------------------
# 5. 保存与显示
# --------------------------
os.makedirs('results', exist_ok=True)

plt.tight_layout()
# 保存为 PDF (矢量图，论文首选) 和 PNG
plt.savefig('results/bucket_average_nmse_ieee.pdf', format='pdf', bbox_inches='tight')
plt.savefig('results/bucket_average_nmse_ieee.png', dpi=300, bbox_inches='tight')

plt.show()