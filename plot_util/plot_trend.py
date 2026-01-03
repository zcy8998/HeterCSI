import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------------------
# 1. IEEE Trans 风格全局设置
# --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,              # 基础字号
    'axes.labelsize': 14,         # 轴标签字号
    'axes.titlesize': 14,         # 标题字号
    'legend.fontsize': 14,        # 图例字号
    'xtick.labelsize': 14,        # X轴刻度字号
    'ytick.labelsize': 14,        # Y轴刻度字号
    'axes.linewidth': 1.0,        # 轴线宽度
    'grid.color': 'gray',
    'grid.linestyle': ':',        # 网格线改为点线，更精致
    'grid.alpha': 0.5,
    'grid.linewidth': 0.5,
    'xtick.direction': 'in',      # 刻度向内 (IEEE风格)
    'ytick.direction': 'in',      # 刻度向内
    'xtick.top': True,            # 上方显示刻度
    'ytick.right': True,          # 右侧显示刻度
    'figure.dpi': 300,            # 高分辨率
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,       # 线宽
    'lines.markersize': 9         # 标记点大小
})

# --------------------------
# 2. 数据录入 (严格对应表格)
# --------------------------
# Dataset Sizes
x_axis = [8, 16, 24, 32, 40]

# Mapping:
# Table 'all' -> 'Global'
# Table 'group(proposed)' -> 'Proposed'
# Table 'seq_new' -> 'Alternating'

# Task 1: Random-0.85
# Data from tables:
# 8:  0.503, 0.464, 0.461
# 16: 0.47,  0.489, 0.408
# 24: 0.42,  0.425, 0.382
# 32: 0.394, 0.342, 0.332
# 40: 0.96,  0.039, 0.1   (Note: 40dataset values show significant variance based on image)
data_random = {
    'Global':      [0.503, 0.47,  0.42,  0.394, 0.096],
    'Proposed':    [0.464, 0.489, 0.425, 0.342, 0.039],
    'Alternating': [0.461, 0.408, 0.382, 0.332, 0.1]
}

# Task 2: Temporal-0.5
# Data from tables:
# 8:  0.647, 0.644, 0.662
# 16: 0.656, 0.576, 0.619
# 24: 0.683, 0.555, 0.638
# 32: 0.577, 0.516, 0.561
# 40: 0.245, 0.161, 0.187 (Labeled just "Temporal" in 40dataset table)
data_temporal = {
    'Global':      [0.647, 0.656, 0.683, 0.577, 0.245],
    'Proposed':    [0.644, 0.576, 0.555, 0.516, 0.161],
    'Alternating': [0.662, 0.619, 0.638, 0.561, 0.187]
}

# Task 3: Freq-0.5
# Data from tables:
# 8:  0.588, 0.562, 0.577
# 16: 0.532, 0.532, 0.525
# 24: 0.557, 0.515, 0.554
# 32: 0.576, 0.534, 0.543
# 40: 0.069, 0.064, 0.85
data_freq = {
    'Global':      [0.588, 0.532, 0.557, 0.576, 0.069],
    'Proposed':    [0.562, 0.532, 0.515, 0.534, 0.064],
    'Alternating': [0.577, 0.525, 0.554, 0.543, 0.085]
}

all_tasks = [
    ("Random Masking (0.85)", data_random),
    ("Temporal Masking (0.5)", data_temporal),
    ("Frequency Masking (0.5)", data_freq)
]

# --------------------------
# 3. 绘图逻辑
# --------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 调整长宽比以适应论文排版

# 定义符合学术规范的样式
styles = {
    'Global':      {'marker': 'o', 'color': '#1f77b4', 'ls': '-',  'label': 'Global'},     # 蓝色实线圆点
    'Proposed':    {'marker': 's', 'color': '#d62728', 'ls': '--', 'label': 'Proposed'},   # 红色虚线方块
    'Alternating': {'marker': '^', 'color': '#2ca02c', 'ls': '-.', 'label': 'Alternating'} # 绿色点划线三角
}

for ax, (task_name, task_data) in zip(axes, all_tasks):
    # 绘制曲线
    for method, values in task_data.items():
        s = styles[method]
        ax.plot(x_axis, values, 
                marker=s['marker'], 
                linestyle=s['ls'], 
                color=s['color'], 
                label=s['label'],
                markeredgecolor='white', # 标记边缘白色，增加对比度
                markeredgewidth=1.5)
    
    # 设置标题与标签
    # ax.set_title(task_name, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Datasets')
    ax.set_ylabel('NMSE')
    
    # X轴刻度设置
    ax.set_xticks(x_axis)
    
    # 网格
    ax.grid(True, which='major', axis='both')
    
    # Y轴范围微调 (可选，防止标签重叠，根据数据自动调整通常即可)
    # ax.set_ylim(bottom=0) 

    # 科学计数法格式化 (如果数值非常小)
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# --------------------------
# 4. 统一图例与布局
# --------------------------
# 获取最后一个子图的句柄和标签，用于生成统一图例
handles, labels = axes[0].get_legend_handles_labels()

# 在图形顶部居中显示图例，一行显示
fig.legend(handles, labels, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 0.95), # 放置在图表上方
           ncol=3,                     # 一行3列
           frameon=False,              # 去除边框
           fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.85) # 留出顶部空间给图例

# --------------------------
# 5. 保存
# --------------------------
plt.savefig('generalization_nmse_ieee_style.png', dpi=300, bbox_inches='tight')
plt.savefig('generalization_nmse_ieee_style.pdf', format='pdf', bbox_inches='tight')

plt.show()