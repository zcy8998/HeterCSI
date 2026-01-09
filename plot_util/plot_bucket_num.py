import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

# --------------------------
# 1. IEEE 风格全局设置 (您提供的配置)
# --------------------------
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
    'grid.color': '#b0b0b0',      # 网格颜色
    'grid.linestyle': '--',       # 网格线虚线
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
# 2. 数据准备 (从 Log 提取)
# --------------------------
bucket_sizes = [1, 2, 4, 8, 16]

# Log 数据 (Unit: dB)
# Random Task
nmse_random =   [-2.2949295, -6.6580005, -12.3381596, -9.5968924, -9.7561750]
# Temporal Task
nmse_temporal = [-7.6857634, -7.8764606, -7.9350448,  -7.9007101, -7.6667280]
# Freq Task
nmse_freq =     [-8.0977650, -7.7854638, -8.3796968,  -8.3352613, -8.2229939]
# Overall Average
nmse_average =  [-5.1575952, -7.4036522, -9.1546164,  -8.5532656, -8.4622364]

# --------------------------
# 3. 样式定义 (您提供的样式字典)
# --------------------------
styles = {
    'Global':      {'marker': 'o', 'color': '#0072BD', 'ls': '--',  'label': 'Global'},     
    'Alternating': {'marker': '^', 'color': '#77AC30', 'ls': '-.', 'label': 'Alternating'}, 
    'Proposed':    {'marker': 's', 'color': '#D95319', 'ls': '-',  'label': 'Proposed'}    
}

# --------------------------
# 4. 绘图逻辑
# --------------------------
fig, ax = plt.subplots(figsize=(6, 5)) 

# 定义数据与样式的映射关系
# 这里假设：Random -> Global, Temporal -> Alternating, Freq -> Proposed
# 如果您的论文中 "Proposed" 指的是 Average 或其他数据，请在这里交换变量
plot_mappings = [
    (nmse_random,   styles['Global']),
    (nmse_temporal, styles['Alternating']),
    (nmse_freq,     styles['Proposed'])
]

# 循环绘图
for data, style in plot_mappings:
    ax.plot(bucket_sizes, data, 
            marker=style['marker'], 
            color=style['color'], 
            linestyle=style['ls'], 
            label=style['label'])

# (可选) 如果您也想画 Average 曲线，可以取消下面代码的注释，并分配一个样式
# ax.plot(bucket_sizes, nmse_average, marker='*', color='k', linestyle='-', label='Average')

# --------------------------
# 5. 坐标轴与细节调整
# --------------------------
ax.set_xlabel('Number of Buckets')
ax.set_ylabel('NMSE (dB)')  # 单位更新为 dB

# 设置 X 轴为 Log Base 2
ax.set_xscale('log', base=2)
ax.set_xticks(bucket_sizes)
ax.get_xaxis().set_major_formatter(ScalarFormatter()) # 显示 1, 2, 4... 而非 2^0, 2^1...

# 网格与图例
ax.grid(True)
# 图例位置自适应，去除边框背景使其更简洁 (framealpha可调)
ax.legend(loc='best', edgecolor='black', fancybox=False, framealpha=1.0)

# --------------------------
# 6. 保存与显示
# --------------------------
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

plt.tight_layout()

# 保存为 PDF (论文首选) 和 PNG
save_path_pdf = os.path.join(output_dir, 'nmse_db_comparison.pdf')
save_path_png = os.path.join(output_dir, 'nmse_db_comparison.png')

plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')

print(f"Figures saved to:\n{save_path_pdf}\n{save_path_png}")
plt.show()