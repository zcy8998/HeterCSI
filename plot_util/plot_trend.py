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
dataset_sizes = [8, 16, 24, 32]

# 任务1: Random-0.85
# 对应表格行: random-0.85
data_random = {
    'Global':      [0.305, 0.136, 0.079, 0.316],
    'Proposed':    [0.259, 0.141, 0.058, 0.039],
    'Alternating': [0.273, 0.195, 0.072, 0.074]
}

# 任务2: Temporal-0.5
# 对应表格行: temporal-0.5
data_temporal = {
    'Global':      [0.375, 0.282, 0.274, 0.2],
    'Proposed':    [0.388, 0.278, 0.246, 0.197],
    'Alternating': [0.377, 0.265, 0.257, 0.208]
}

# 任务3: Freq-0.5
# 对应表格行: freq-0.5
data_freq = {
    'Global':      [0.286, 0.162, 0.136, 0.093],
    'Proposed':    [0.258, 0.135, 0.126, 0.095],
    'Alternating': [0.273, 0.176, 0.135, 0.1]
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
                linewidth=2,         # 线宽参考值为2
                markersize=8)        # 标记大小参考值为8
    
    # 样式修饰
    # ax.set_title(task_name)           # 不显示标题
    ax.set_xlabel('Number of Datasets') # 修改横轴标签
    ax.set_ylabel('NMSE')
    ax.set_xticks(dataset_sizes)        # 强制显示离散的横轴刻度
    ax.tick_params(axis='both', which='major') 
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 图例
    ax.legend(loc='best', frameon=True)

plt.tight_layout()

# --------------------------
# 4. 保存图片
# --------------------------
plt.savefig('generalization_nmse_result.png', dpi=300, bbox_inches='tight')
plt.savefig('generalization_nmse_result.pdf', format='pdf', bbox_inches='tight')

plt.show()
