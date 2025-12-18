import numpy as np
import matplotlib.pyplot as plt

# 设置科研风格参数
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'lightgrey',
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})

# 数据设置 - 假设两种条件：Condition A 和 Condition B
conditions = ['D1', 'D1-D4']
schemes = ['Proposed', 'Sequential']

# 数据矩阵：行表示条件，列表示方案
# time
data = np.array([
    [0.068, 1.12],  # Condition A: Proposed, Bucket
    [0.072, 0.536]   # Condition B: Proposed, Bucket
])
# fre
# data = np.array([
#     [0.271, 1.82],  # Condition A: Proposed, Bucket
#     [0.072, 0.723]   # Condition B: Proposed, Bucket
# ])
# recon
# data = np.array([
#     [0.024, 1.12],  # Condition A: Proposed, Bucket
#     [0.019, 0.545]   # Condition B: Proposed, Bucket
# ])

# 方案颜色 - 每种方案固定颜色
scheme_colors = {
    'Proposed': '#2c7bb6',  # 深蓝色
    'Sequential': '#94b6d1'     # 浅蓝色
}

# 创建图表
fig, ax = plt.subplots(figsize=(7, 6))
fig.set_facecolor('white')
ax.set_facecolor('white')

# 设置柱状图参数
n_conditions = len(conditions)
n_schemes = len(schemes)
bar_width = 0.35  # 单个柱子宽度
group_width = bar_width * n_schemes  # 每组总宽度
group_spacing = 0.2  # 组间间距

# 计算每组起始位置
x = np.arange(n_conditions) * (group_width + group_spacing)

# 绘制柱状图
bars = []
for i, scheme in enumerate(schemes):
    # 计算当前方案在每组中的位置
    x_pos = x + i * bar_width
    bar = ax.bar(x_pos, data[:, i], bar_width,
                 color=scheme_colors[scheme],
                 edgecolor='black',
                 linewidth=1.0,
                 alpha=0.95,
                 label=scheme)
    bars.append(bar)
    
    # 添加柱顶数值
    for j, val in enumerate(data[:, i]):
        ax.annotate(f'{val:.2f}',
                    xy=(x_pos[j], val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)

# 设置坐标轴
ax.set_xlabel('Temporal-domain Prediction', fontsize=16, labelpad=15)
ax.set_ylabel('Normalized MSE (NMSE)', fontsize=16, labelpad=15)
ax.set_xticks(x + group_width/2 - bar_width/2)
ax.set_xticklabels(conditions, fontsize=15)
ax.tick_params(axis='x', which='major', pad=10)
ax.tick_params(axis='y', labelsize=14)

# 添加网格线
ax.grid(True, axis='y', alpha=0.6)
ax.grid(False, axis='x')

# 添加图例
ax.legend(fontsize=12, frameon=True, loc='upper right')

# 确保坐标轴和标签清晰
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)

# 调整布局
plt.tight_layout(pad=2.0)

# 保存高质量科研图像
plt.savefig('results_fig/forget_time.png', dpi=300, bbox_inches='tight')
plt.show()