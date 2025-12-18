import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker

# --------------------------
# 1. 配置区域
# --------------------------

# 映射名称和路径
# 策略123 -> Global/Group/Sequential
experiment_configs = {
    "Global": {
        "path": "/data/zcy_new/cross_csi/global_motivation_v1",
        "color": "#1f77b4", # 蓝
    },
    "Group": {
        "path": "/data/zcy_new/cross_csi/bucket_motivation_v1",
        "color": "#d62728", # 红
    },
    "Sequential": {
        "path": "/data/zcy_new/cross_csi/seq_motivation_v1",
        "color": "#2ca02c", # 绿
    }
}

# 结果保存目录
save_dir = Path("results")
save_dir.mkdir(parents=True, exist_ok=True)

# 检查点范围
max_ckpt_num = 100

# --------------------------
# 2. 科研绘图风格设置
# --------------------------
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 14,             # 整体字体稍微调大
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.5,       # 坐标轴线变粗
    "grid.alpha": 0.5,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "figure.dpi": 300,
    "savefig.bbox": 'tight',
}
plt.rcParams.update(config)

# --------------------------
# 3. 数据读取 (预加载)
# --------------------------

subdirs = ['test1', 'test2']
# 缓存结构: cache[subdir][method_name][metric_type][epoch] = value
data_cache = {sd: {name: {'all': {}, 'temporal': {}} for name in experiment_configs} for sd in subdirs}

print("Reading data...")

for subdir in subdirs:
    for method_name, config in experiment_configs.items():
        base_path = config["path"]
        dir_path = Path(base_path) / subdir
        
        # 简单进度提示
        # print(f"Processing: [{subdir}] {method_name}...")
        
        if not dir_path.exists():
            continue

        for ckpt_num in range(max_ckpt_num):
            file_path = dir_path / f"nmse_{ckpt_num}.pth"
            if not file_path.exists():
                continue
            try:
                results = torch.load(file_path, map_location='cpu', weights_only=False)
                if 'all' in results and 'avg_nmse' in results['all']:
                    data_cache[subdir][method_name]['all'][ckpt_num] = results['all']['avg_nmse']
                if 'temporal' in results and 'avg_nmse' in results['temporal']:
                    data_cache[subdir][method_name]['temporal'][ckpt_num] = results['temporal']['avg_nmse']
            except:
                pass

print("Data reading completed.")

# --------------------------
# 4. 绘图 (宽比例 + 底部标题)
# --------------------------

# 设置画布大小：宽14，高9 (长方形，非正方形)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# 定义子图配置
# Test 1 -> In-Domain Test
# Test 2 -> Zero-Shot Test
plot_configs = [
    (0, 0, 'test1', 'all',      '(a) In-Domain Test: Overall NMSE'),
    (0, 1, 'test1', 'temporal', '(b) In-Domain Test: Temporal NMSE'),
    (1, 0, 'test2', 'all',      '(c) Zero-Shot Test: Overall NMSE'),
    (1, 1, 'test2', 'temporal', '(d) Zero-Shot Test: Temporal NMSE')
]

for row, col, subdir, metric_key, title_text in plot_configs:
    ax = axes[row, col]
    
    all_x_in_plot = []
    
    for method_name, config in experiment_configs.items():
        data_dict = data_cache[subdir][method_name].get(metric_key, {})
        
        if not data_dict:
            continue
            
        x_values = sorted(data_dict.keys())
        y_values = [data_dict[k] for k in x_values]
        all_x_in_plot.extend(x_values)
        
        # 绘图: 全实线，无Marker
        ax.plot(x_values, y_values, 
                label=method_name,
                color=config['color'],
                linestyle='-',    # 全部实线
                linewidth=2.5,    # 线条稍微加粗
                alpha=0.95)

    # --- 标题设置 (移到底部) ---
    # y坐标设置为负数，使其位于x轴标签下方
    # 注意：根据实际画面可能需要微调 -0.25 这个数值
    ax.set_title(title_text, y=-0.26, fontweight='bold', fontsize=16)

    # --- 轴标签 ---
    ax.set_xlabel('Training Epochs')
    
    if col == 0:
        ax.set_ylabel('NMSE')
    else:
        ax.set_ylabel('') # 只保留刻度，不重复Label

    # --- 网格与刻度 ---
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    
    if all_x_in_plot:
        ax.set_xlim(0, max(all_x_in_plot) + 1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    # --- 图例设置 ---
    # 仅在第一个图显示图例，或每个都显示 (这里每个都显示，但去掉边框)
    ax.legend(loc='upper right', frameon=False, fontsize=13)

# --------------------------
# 5. 布局调整与保存
# --------------------------

# 自动调整布局，但留出底部空间给标题
# hspace=0.4 是为了防止上面图的 Training Epochs 和下面图的 Title 撞车
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.15, hspace=0.45)

output_filename = "nmse_comparison_wide_solid_bottom_title"
pdf_path = save_dir / f"{output_filename}.pdf"
png_path = save_dir / f"{output_filename}.png"

fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"\nPlotting Finished!")
print(f"PDF saved to: {pdf_path}")
print(f"PNG saved to: {png_path}")
