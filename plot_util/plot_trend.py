import matplotlib.pyplot as plt

# --------------------------
# 1. 字体与全局设置 (保持原风格)
# --------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,              # 全局字号控制
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'lightgrey',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'legend.frameon': False,      # 图例边框控制
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# --------------------------
# 2. 数据准备 (已根据表格更新)
# --------------------------
# 对应表格中的 8dataset, 16dataset, 24dataset, 32dataset, 40dataset
dataset_sizes = [8, 16, 24, 32, 40]

# 映射关系:
# 表格 'all'             -> 代码 'Global'
# 表格 'group(proposed)' -> 代码 'Proposed'
# 表格 'seq_new'         -> 代码 'Alternating'

# 任务1: Random-0.85
# 数据来源: 对应各表格 random-0.85 行
data_random = {
    'Global':      [0.505, 0.378, 0.317, 0.414, 0.141],
    'Proposed':    [0.438, 0.341, 0.299, 0.252, 0.106],
    'Alternating': [0.445, 0.321, 0.264, 0.266, 0.172]
}

# 任务2: Temporal-0.5
# 数据来源: 对应各表格 temporal-0.5 行
data_temporal = {
    'Global':      [0.6,   0.493, 0.51,  0.463, 0.315],
    'Proposed':    [0.596, 0.442, 0.456, 0.429, 0.284],
    'Alternating': [0.609, 0.484, 0.494, 0.469, 0.323]
}

# 任务3: Freq-0.5
# 数据来源: 对应各表格 freq-0.5 行
data_freq = {
    'Global':      [0.462, 0.355, 0.353, 0.358, 0.168],
    'Proposed':    [0.435, 0.341, 0.334, 0.339, 0.13],
    'Alternating': [0.444, 0.34,  0.349, 0.35,  0.15]
}

# 整合任务列表
all_tasks = [
    ("Random-0.85", data_random),
    ("Temporal-0.5", data_temporal),
    ("Freq-0.5", data_freq)
]

# --------------------------
# 3. 绘图逻辑
# --------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

# 定义曲线样式
styles = {
    'Global':      {'marker': 'o', 'linestyle': '-',  'color': '#1f77b4', 'label': 'Global'},      # 蓝色实线
    'Proposed':    {'marker': 's', 'linestyle': '--', 'color': '#d62728', 'label': 'Proposed'},    # 红色虚线
    'Alternating': {'marker': 'D', 'linestyle': '-.', 'color': '#2ca02c', 'label': 'Alternating'}  # 绿色点划线
}

for ax, (task_name, task_data) in zip(axes, all_tasks):
    # 绘制曲线
    for method_key, values in task_data.items():
        style = styles[method_key]
        ax.plot(dataset_sizes, values, 
                marker=style['marker'], 
                linestyle=style['linestyle'], 
                color=style['color'], 
                label=style['label'],
                linewidth=2,         # 线宽
                markersize=8)        # 标记大小
    
    # 样式修饰
    # ax.set_title(task_name)           # 标题(可选)
    ax.set_xlabel('Number of Datasets') # 横轴标签
    ax.set_ylabel('NMSE')
    ax.set_xticks(dataset_sizes)        # 强制显示离散的横轴刻度 (8, 16, 24, 32, 40)
    ax.tick_params(axis='both', which='major') 
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 图例
    ax.legend(loc='best', frameon=True)

plt.tight_layout()

# --------------------------
# 4. 保存图片
# --------------------------
plt.savefig('results/generalization_nmse_result.png', dpi=300, bbox_inches='tight')
plt.savefig('results/generalization_nmse_result.pdf', format='pdf', bbox_inches='tight')

plt.show()