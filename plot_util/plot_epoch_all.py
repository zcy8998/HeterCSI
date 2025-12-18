import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------
# 1. 配置区域 (请修改这里)
# --------------------------

# 定义三个不同的路径和对应的图例名称
experiment_paths = {
    "Strategy 1": "/data/zcy_new/cross_csi/global_motivation_v1/test2",
    "Strategy 2": "/data/zcy_new/cross_csi/bucket_motivation_v1/test2",
    "Strategy 3": "/data/zcy_new/cross_csi/seq_motivation_v1/test2"
}

# 结果保存目录
save_dir = Path("results")
save_dir.mkdir(parents=True, exist_ok=True)

# 检查点范围 (例如从 nmse_0.pth 到 nmse_54.pth)
max_ckpt_num = 100

# --------------------------
# 2. 科研绘图风格设置
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
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

COLORS = [
    '#1f77b4', '#d62728', '#2ca02c', # 蓝, 红, 绿 (主要对比色)
    '#ff7f0e', '#9467bd', '#8c564b'
]

# --------------------------
# 3. 数据读取
# --------------------------

# 存储结构: { "Method A": {ckpt_num: nmse_val}, ... }
all_data = {name: {} for name in experiment_paths.keys()}

print("开始读取数据...")

for exp_name, dir_path in experiment_paths.items():
    dir_path = Path(dir_path)
    print(f"正在处理: {exp_name} -> {dir_path}")
    
    valid_count = 0
    for ckpt_num in range(max_ckpt_num):
        file_path = dir_path / f"nmse_{ckpt_num}.pth"
        
        if not file_path.exists():
            continue
            
        try:
            # 加载 .pth 文件
            results = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # 提取 key 为 'all' 的 avg_nmse
            if 'all' in results and 'avg_nmse' in results['all']:
                nmse_val = results['all']['avg_nmse']
                all_data[exp_name][ckpt_num] = nmse_val
                valid_count += 1
            else:
                print(f"  警告: {file_path.name} 中缺少 ['all']['avg_nmse'] 字段")
                
        except Exception as e:
            print(f"  读取错误 {file_path}: {e}")
            
    print(f"  - 成功读取 {valid_count} 个数据点")

# --------------------------
# 4. 绘图
# --------------------------

fig, ax = plt.subplots(figsize=(8, 7))

# 收集所有出现的 checkpoint 编号以确定 X 轴范围
all_ckpts = set()
for data in all_data.values():
    all_ckpts.update(data.keys())

if not all_ckpts:
    raise ValueError("没有读取到任何有效数据，请检查路径和文件名。")

sorted_all_ckpts = sorted(list(all_ckpts))

# 遍历每个实验进行绘图
for i, (exp_name, data_dict) in enumerate(all_data.items()):
    if not data_dict:
        print(f"跳过空数据: {exp_name}")
        continue

    # 整理 x 和 y 数据 (确保按 epoch 排序)
    x_values = sorted(data_dict.keys())
    y_values = [data_dict[k] for k in x_values]
    
    color = COLORS[i % len(COLORS)]
    
    ax.plot(x_values, y_values, 
            color=color, 
            linewidth=2.5,
            markersize=6,
            markeredgewidth=1.5,
            markeredgecolor='white',
            markerfacecolor=color,
            label=exp_name,
            zorder=10)

# 设置图表属性
ax.set_xlabel('Training Epochs', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('Average NMSE', fontsize=16, fontweight='bold', labelpad=10)
# ax.set_title('Performance Comparison', fontsize=18, fontweight='bold', pad=20)

# 网格和刻度
ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
ax.tick_params(axis='both', which='major', labelsize=14, width=1.2, length=6)

# 动态设置 X 轴范围
if sorted_all_ckpts:
    ax.set_xlim(min(sorted_all_ckpts), max(sorted_all_ckpts))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # 强制整数刻度

# 图例
legend = ax.legend(loc='upper right', 
                   fontsize=14,
                   frameon=False)

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# --------------------------
# 5. 保存
# --------------------------
output_filename = "nmse_comparison_all"
pdf_path = save_dir / f"{output_filename}.pdf"
png_path = save_dir / f"{output_filename}.png"

fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"\n绘图完成！")
print(f"PDF已保存至: {pdf_path}")
print(f"PNG已保存至: {png_path}")
