import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================
# 1. 科研风格全局设置 (Times New Roman)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'grid.color': 'lightgrey',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'figure.dpi': 300
})

# ==========================================
# 2. 数据录入
# ==========================================
schemes = ['Proposed', 'WiFo', 'LLM4CP', 'Transformer', 'LSTM', 'PAD']

# 原始数据
raw_data = {
    'zero': [0.03, 0.17, np.nan, np.nan, np.nan, 2.038],
    'few':  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'full': [np.nan, np.nan, 0.038, 0.059, 0.085, np.nan]
}

# 绘图配置映射
configs = {
    'zero': {'label': 'Zero-shot', 'color': '#2c7bb6', 'hatch': None},
    'few':  {'label': 'Few-shot',  'color': '#7fcdbb', 'hatch': '///'},
    'full': {'label': 'Full-shot', 'color': '#fdae61', 'hatch': '\\\\'}
}

# 顺序定义 (用于决定左中右的相对顺序)
category_order = ['zero', 'few', 'full']

# ==========================================
# 3. 动态坐标计算 (核心修改部分)
# ==========================================
bar_width = 0.3  # 柱子宽度

# 初始化用于绘图的坐标字典
# 结构: {'zero': [x1, nan, x3...], 'few': [x1, nan, ...]}
plot_positions = {cat: [np.nan] * len(schemes) for cat in category_order}

for i in range(len(schemes)):
    # 1. 找出当前 scheme (第i组) 有哪些有效数据
    valid_cats = []
    for cat in category_order:
        if not np.isnan(raw_data[cat][i]):
            valid_cats.append(cat)
    
    n = len(valid_cats)
    if n == 0:
        continue
        
    # 2. 计算居中偏移量
    # 公式: (index - (n-1)/2) * width
    # n=1 -> 偏移 0
    # n=2 -> 偏移 -0.5w, +0.5w
    # n=3 -> 偏移 -1w, 0, +1w
    offsets = [(k - (n - 1) / 2) * bar_width for k in range(n)]
    
    # 3. 将计算出的绝对x坐标填入对应的列表
    for k, cat in enumerate(valid_cats):
        plot_positions[cat][i] = i + offsets[k]

# ==========================================
# 4. 绘图逻辑
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

# 遍历每一类 (Zero, Few, Full) 进行绘制
for cat in category_order:
    x_coords = plot_positions[cat]
    y_values = raw_data[cat]
    style = configs[cat]
    
    # 过滤掉 NaN 的点进行绘制 (ax.bar 需要具体的 x 和 height)
    # 我们只绘制该类别存在的那些柱子
    clean_x = []
    clean_y = []
    for x, y in zip(x_coords, y_values):
        if not np.isnan(x) and not np.isnan(y):
            clean_x.append(x)
            clean_y.append(y)
            
    if not clean_x:
        continue

    # 绘制柱状图
    rects = ax.bar(clean_x, clean_y, width=bar_width,
                   label=style['label'], 
                   color=style['color'], 
                   edgecolor='black',
                   linewidth=0.8, 
                   hatch=style['hatch'], 
                   alpha=0.9, 
                   zorder=3)
    
    # 添加数值标签
    for rect, val in zip(rects, clean_y):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, color='black')

# ==========================================
# 5. 坐标轴与美化
# ==========================================
ax.set_ylabel('NMSE', fontsize=14, fontweight='bold')
# ax.set_xlabel('Schemes', fontsize=14, fontweight='bold')

ax.set_xticks(range(len(schemes)))
ax.set_xticklabels(schemes, fontsize=12)

# 动态调整 Y 轴上限
all_values = raw_data['zero'] + raw_data['few'] + raw_data['full']
max_val = np.nanmax(all_values)
ax.set_ylim(0, max_val * 1.15)

# 网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)

# 图例设置 (自动去重)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98),
          ncol=3, frameon=True, edgecolor='black', fancybox=False, fontsize=11)

plt.tight_layout()

# 保存
plt.savefig('results/comparison_result_centered.pdf', format='pdf', bbox_inches='tight')
plt.savefig('results/comparison_result_centered.png', format='png', bbox_inches='tight', dpi=300)

plt.show()
