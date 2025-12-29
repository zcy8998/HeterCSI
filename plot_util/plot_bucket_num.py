import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------
# 1. 字体与全局设置
# --------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'lightgrey',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'legend.frameon': True,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# --------------------------
# 2. 数据准备
# --------------------------
bucket_sizes = [1, 4, 8, 12, 16]

# 原始数据
nmse_random = [0.141, 0.106, 0.119, 0.121, 0.111]
nmse_temporal = [0.315, 0.284, 0.296, 0.309, 0.303]
nmse_freq = [0.168, 0.130, 0.138, 0.121, 0.116]

# 使用 numpy 计算平均值
data_matrix = np.array([nmse_random, nmse_temporal, nmse_freq])
nmse_average = np.mean(data_matrix, axis=0)  # 按列求平均

# --------------------------
# 3. 绘图逻辑
# --------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# --- A. 绘制背景线 (原始任务) ---
# 用半透明(alpha=0.3)及细线显示，作为参考背景
ax.plot(bucket_sizes, nmse_random, marker='o', linestyle='-', 
        color='#1f77b4', alpha=0.3, linewidth=1, label='Random-0.85')
ax.plot(bucket_sizes, nmse_temporal, marker='s', linestyle='--', 
        color='#d62728', alpha=0.3, linewidth=1, label='Temporal-0.5')
ax.plot(bucket_sizes, nmse_freq, marker='D', linestyle='-.', 
        color='#2ca02c', alpha=0.3, linewidth=1, label='Freq-0.5')

# --- B. 绘制平均趋势线 (重点) ---
# 使用黑色、加粗、不同标记突出显示
ax.plot(bucket_sizes, nmse_average, 
        marker='*',          # 星号标记
        linestyle='-',       # 实线
        color='black',       # 黑色
        linewidth=3,         # 线宽加粗
        markersize=12,       # 标记加大
        label='Average Trend') # 图例

# --------------------------
# 4. 样式修饰
# --------------------------
ax.set_xlabel('Number of Buckets')
ax.set_ylabel('Average NMSE')  # 纵轴标签改为 Average NMSE
ax.set_xticks(bucket_sizes)

ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', which='major')

# 图例设置 (将 Average 排在第一位或显眼位置)
# 这里使用 handles, labels 重新排序图例，确保 Average 在最上面
handles, labels = ax.get_legend_handles_labels()
# 重新排序：把最后一个(Average)移到第一个
order = [3, 0, 1, 2] 
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
          loc='best', frameon=True, edgecolor='black', fancybox=False)

# --------------------------
# 5. 保存与显示
# --------------------------
os.makedirs('results', exist_ok=True)

plt.tight_layout()
plt.savefig('results/bucket_average_nmse.png', dpi=300, bbox_inches='tight')
plt.savefig('results/bucket_average_nmse.pdf', format='pdf', bbox_inches='tight')

plt.show()