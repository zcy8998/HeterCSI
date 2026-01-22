import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --------------------------
# 1. IEEE Trans 风格全局设置
# --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,              # 字体大小适中
    'axes.linewidth': 1.0,        # 轴线宽度
    'grid.color': '#b0b0b0',      # 网格颜色
    'grid.linestyle': ':',        # 网格线为点线
    'grid.alpha': 0.6,
    'xtick.direction': 'in',      # 刻度向内
    'ytick.direction': 'in',      # 刻度向内
    'xtick.top': True,            # 上方显示刻度
    'ytick.right': True,          # 右侧显示刻度
    'figure.dpi': 300,            # 高分辨率
    'savefig.bbox': 'tight',      # 去除白边
    'lines.linewidth': 1.5,       # 线宽
    'lines.markersize': 7         # 标记点大小
})

# --------------------------
# 2. 数据准备
# --------------------------
buckets = [1, 2, 4, 8, 16]

# --- NMSE Data (Left Axis) ---
# High Heterogeneity (NMSE)
nmse_high = [
    -5.1575952, -7.4036522, -9.1546164, -8.5532656, -8.4622364
]
# Low Heterogeneity (NMSE)
nmse_low  = [
    -2.4285100, -2.7215242, -2.5983577, -2.5961118, -2.4479625
]

# --- Time Data (Right Axis, from Table) ---
# High Heterogeneity (Time)
time_high = [12.82, 8.11, 5.95, 5.07, 4.53]

# Low Heterogeneity (Time)
time_low  = [3.21, 3.23, 3.21, 3.21, 3.20]

# --------------------------
# 3. 绘图逻辑 (双Y轴)
# --------------------------
fig, ax1 = plt.subplots(figsize=(7, 6))

# 创建共享X轴的第二个Y轴
ax2 = ax1.twinx()

# 定义颜色
color_high = '#0072BD'  # 蓝色
color_low  = '#D95319'  # 橙红色

# --- 绘制左轴 (NMSE) - 实线, 实心标记 ---
l1 = ax1.plot(buckets, nmse_high, 
              color=color_high, marker='o', linestyle='-', 
              label='High Het. (NMSE)', zorder=10)
l2 = ax1.plot(buckets, nmse_low, 
              color=color_low, marker='s', linestyle='-', 
              label='Low Het. (NMSE)', zorder=10)

# --- 绘制右轴 (Time) - 虚线, 空心标记 ---
# markerfacecolor='white' 制造空心效果
l3 = ax2.plot(buckets, time_high, 
              color=color_high, marker='o', linestyle='--', 
              markerfacecolor='white', markeredgewidth=1.5,
              label='High Het. (Time)', zorder=10)
l4 = ax2.plot(buckets, time_low, 
              color=color_low, marker='s', linestyle='--', 
              markerfacecolor='white', markeredgewidth=1.5,
              label='Low Het. (Time)', zorder=10)

# --------------------------
# 4. 坐标轴设置
# --------------------------

# --- X轴设置 ---
ax1.set_xlabel('Number of Buckets')
ax1.set_xscale('log', base=2)
ax1.set_xticks(buckets)
ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

# --- 左Y轴设置 (NMSE) ---
ax1.set_ylabel('NMSE (dB)')
ax1.set_ylim(-11, -1)  # 根据数据范围调整
ax1.grid(True)         # 仅在主轴显示网格，避免混乱

# --- 右Y轴设置 (Time) ---
ax2.set_ylabel('Training Time (min)')
ax2.set_ylim(0, 14)    # 适应 12.34 的最大值
# 可以在右轴略微调整刻度位置，但通常不建议双重网格

# --------------------------
# 5. 合并图例
# --------------------------
# 获取两个轴的句柄和标签，合并到一个图例框中
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

lines = lines_1 + lines_2
labels = labels_1 + labels_2

# 图例位置建议放在“中间右侧”或“上方居中”，避开曲线
ax1.legend(lines, labels, loc='center right', 
           frameon=True, edgecolor='black', framealpha=1, 
           bbox_to_anchor=(1.0, 0.6)) # 微调位置

# --------------------------
# 6. 保存与显示
# --------------------------
plt.tight_layout()
plt.savefig('bucket_dual_axis.pdf', format='pdf')
plt.savefig('bucket_dual_axis.png', dpi=300)
plt.show()