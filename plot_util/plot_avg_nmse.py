import numpy as np
import matplotlib.pyplot as plt

# 设置科研风格参数
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,  # 增大基础字号
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'lightgrey',
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})

# 数据设置
schemes = ['All', 'Proposed', 'Bucket', 'Sequential', ]
zero_shot = [0.25, 0.29, 0.92, 1.20]

# 新的蓝色/灰色调配色方案
colors = [
    '#2c7bb6',  # Proposed - 保持深蓝色以突出重点
    '#5a9ac7',  # WiFo - 中蓝色
    '#94b6d1',  # Transformer - 蓝灰色
    '#b0c9df',  # LSTM - 浅蓝灰色
    '#d0d9e8',  # 3D ResNet - 淡蓝灰色
    '#dbe1e8',  # PAD - 冷灰色
    '#e4e8ef'   # LLM4CP - 浅灰色
]

x = np.arange(len(schemes))
bar_width = 0.45  # 缩小柱子宽度

# 创建图表
fig, ax = plt.subplots(figsize=(11, 6))
fig.set_facecolor('white')
ax.set_facecolor('white')

# 绘制柱状图
bars = ax.bar(x, zero_shot, bar_width,
              color=colors, edgecolor='black',
              linewidth=1.0, alpha=0.95)

# 添加柱顶数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

# 设置坐标轴
ax.set_xlabel('Time-domain Channel Prediction Schemes',
              fontsize=16, labelpad=15, fontweight='normal')
ax.set_ylabel('Normalized MSE (NMSE)',
              fontsize=16, labelpad=15, fontweight='normal')
ax.set_xticks(x)
ax.set_xticklabels(schemes, fontsize=15, rotation=0)
# ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
# ax.set_ylim(0, 2.05)
ax.tick_params(axis='x', which='major', pad=10)
ax.tick_params(axis='y', labelsize=14)

# 添加网格线
ax.grid(True, axis='y', alpha=0.6)
ax.grid(False, axis='x')

# 确保坐标轴和标签清晰
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)

# 调整布局
plt.tight_layout(pad=2.0)

# 保存高质量科研图像
plt.savefig('results_fig/avg_nmse.png', dpi=300, bbox_inches='tight')
plt.show()