import matplotlib.pyplot as plt

# --------------------------
# 1. IEEE Trans 风格全局设置
# --------------------------
# 设置字体为 Times New Roman，符合 IEEE 标准
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,              # 基础字号
    'axes.labelsize': 12,         # 轴标签字号
    'axes.titlesize': 12,         # 标题字号
    'legend.fontsize': 11,        # 图例字号
    'xtick.labelsize': 11,        # X轴刻度字号
    'ytick.labelsize': 11,        # Y轴刻度字号
    'axes.linewidth': 1.0,        # 轴线宽度
    'grid.color': '#b0b0b0',      # 网格颜色略深一点，保证可见性
    'grid.linestyle': '--',       # 网格线改为虚线
    'grid.alpha': 0.6,
    'grid.linewidth': 0.5,
    'xtick.direction': 'in',      # 刻度向内
    'ytick.direction': 'in',      # 刻度向内
    'xtick.top': True,            # 上方显示刻度
    'ytick.right': True,          # 右侧显示刻度
    'figure.dpi': 300,            # 高分辨率
    'lines.linewidth': 1.5,       # 线宽
    'lines.markersize': 6         # 标记点大小
})

# --------------------------
# 2. 数据录入 (根据图片表格精确提取)
# --------------------------
# Dataset Sizes
x_axis = [8, 16, 24, 32, 40]

# Task 1: Random-0.85
# Data extracted from tables
data_random = {
    'Global':      [-5.8, -7.14, -7.51, -6.68, -12.15],
    'Proposed':    [-6.17, -7.17, -7.73, -8.47, -14.74],
    'Alternating': [-6.17, -7.06, -7.28, -8.39, -10.23]
}

# Task 2: Temporal-0.5
# Data extracted from tables
data_temporal = {
    'Global':      [-4.1, -4.49, -4.47, -5.18, -7.19],
    'Proposed':    [-4.16, -4.98, -5.20, -5.45, -8.68],
    'Alternating': [-4.03, -4.68, -4.65, -5.14, -8.06]
}

# Task 3: Freq-0.5
# Data extracted from tables
data_freq = {
    'Global':      [-4.13, -4.93, -4.84, -4.87, -9.17],
    'Proposed':    [-4.23, -4.99, -5.08, -5.08, -9.34],
    'Alternating': [-4.19, -4.92, -4.70, -4.89, -8.40]
}

all_tasks = [
    ("CSI reconstruction", data_random),
    ("Time-domain prediction", data_temporal),
    ("Frequency-domain prediction", data_freq)
]

# --------------------------
# 3. 绘图逻辑
# --------------------------
# figsize 设置为适合双栏排版的一栏宽度 (约 7-8 英寸) 或全宽
fig, axes = plt.subplots(1, 3, figsize=(14, 5)) 

# 定义样式：
# Proposed: 红色实线，实心方块 (最显眼)
# Global: 蓝色虚线，实心圆
# Alternating: 绿色点划线，实心三角
styles = {
    'Global':      {'marker': 'o', 'color': '#0072BD', 'ls': '--',  'label': 'Global'},     
    'Alternating': {'marker': '^', 'color': '#77AC30', 'ls': '-.', 'label': 'Alternating'}, 
    'Proposed':    {'marker': 's', 'color': '#D95319', 'ls': '-',  'label': 'Proposed'}    
}

for ax, (task_name, task_data) in zip(axes, all_tasks):
    # 绘制曲线
    for method, values in task_data.items():
        s = styles[method]
        
        # Proposed 的 zorder 设置高一些，确保画在最上面
        z_ord = 10 if method == 'Proposed' else 5
        lw = 2.0 if method == 'Proposed' else 1.5 # 稍微加粗 Proposed
        
        ax.plot(x_axis, values, 
                marker=s['marker'], 
                linestyle=s['ls'], 
                color=s['color'], 
                label=s['label'],
                linewidth=lw,
                markeredgecolor=s['color'],
                markerfacecolor='none', # 空心标记点，显得更干净，也可以改成 s['color'] 实心
                markeredgewidth=1.5,
                zorder=z_ord) 
    
    # 标题
    ax.set_title(task_name, fontweight='bold', pad=10)
    
    # 坐标轴标签
    ax.set_xlabel('Number of Datasets')
    if ax == axes[0]: # 只在第一个图显示 Y 轴标签
        ax.set_ylabel('NMSE (dB)')
    
    # X轴刻度设置
    ax.set_xticks(x_axis)
    
    # 网格
    ax.grid(True)

# --------------------------
# 4. 统一图例与布局
# --------------------------
# 获取句柄和标签 (提取 Proposed, Global, Alternating 的顺序)
handles, labels = axes[0].get_legend_handles_labels()

# 调整图例顺序，如果你想让 Proposed 排在第一个，可以在这里手动排序
# 这里演示按照 styles 字典定义的顺序或者绘图顺序
# 为了美观，我们把 Proposed 放中间或第一个通常更好，这里保持默认提取顺序

fig.legend(handles, labels, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.0), # 放在画布最上方
           ncol=3,                     # 横向排列
           frameon=False,              # 无边框图例
           columnspacing=2.0)          # 图例间距

plt.tight_layout()
# 再次调整顶部边距，防止图例重叠
plt.subplots_adjust(top=0.85) 

# --------------------------
# 5. 保存
# --------------------------
# 保存为 PDF (矢量图，论文首选) 和 PNG (预览)
plt.savefig('generalization_nmse_dB_plot.pdf', format='pdf', bbox_inches='tight')
plt.savefig('generalization_nmse_dB_plot.png', dpi=300, bbox_inches='tight')

plt.show()