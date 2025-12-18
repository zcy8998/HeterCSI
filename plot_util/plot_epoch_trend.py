import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 设置科研绘图风格
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
    'legend.frameon': False,  # 无框图例
    'figure.dpi': 300,        # 高分辨率
    'savefig.bbox': 'tight',  # 紧凑保存
    'savefig.pad_inches': 0.1
})

COLORS = [
    '#1f77b4',  # 深蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 橄榄绿
    '#17becf'   # 青色
]

output_dir = "/home/zhangchenyu/data/CSIGPT/pretrain_csi_pretrain_3mask_newbucket256_lr8_16data/test2"
# 获取所有存在的任务类型（从第一个有效文件中获取）
task_types = set()
for ckpt_num in range(0, 2):
    file_path = Path(output_dir) / f"nmse_{ckpt_num}.pth"
    if file_path.exists():
        results = torch.load(file_path, weights_only=False)
        task_types = set(results.keys())
        break

print(f"检测到任务类型: {sorted(task_types)}")

# 初始化数据存储结构
# {task_type: {ckpt_num: avg_nmse}}
all_results = {task: {} for task in task_types}

# 读取所有检查点文件
print("正在读取结果文件...")
valid_checkpoints = []
for ckpt_num in range(0, 55):
    file_path = Path(output_dir) / f"nmse_{ckpt_num}.pth"
    
    if not file_path.exists():
        print(f"跳过不存在的文件: {file_path}")
        continue
        
    try:
        results = torch.load(file_path, weights_only=False)
        valid_checkpoints.append(ckpt_num)
        
        # 提取每个任务的平均NMSE
        for task in task_types:
            if task in results:
                all_results[task][ckpt_num] = results[task]['avg_nmse']
            else:
                print(f"警告: 任务 '{task}' 在 checkpoint {ckpt_num} 中不存在")
                
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")

if not valid_checkpoints:
    raise ValueError("未找到任何有效结果文件")

print(f"成功读取 {len(valid_checkpoints)} 个检查点文件")

# 准备绘图数据
x_values = sorted(valid_checkpoints)

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 为每个任务绘制曲线
for i, task in enumerate(sorted(task_types)):
    y_values = []
    for ckpt in x_values:
        # 获取该检查点的NMSE，若不存在则用NaN
        y_values.append(all_results[task].get(ckpt, np.nan))
    
    # 使用色盲友好配色
    color = COLORS[i % len(COLORS)]
    
    # 绘制带标记的曲线
    ax.plot(x_values, y_values, 
            color=color, 
            linewidth=2.5,
            marker='o', 
            markersize=6,
            markeredgewidth=1.5,
            markeredgecolor='white',
            markerfacecolor=color,
            label=f"{task.capitalize()}",
            zorder=10)  # 确保曲线在网格线上方

# 设置图表属性
ax.set_xlabel('Training Epochs', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('Average NMSE', fontsize=16, fontweight='bold', labelpad=10)
ax.set_title('NMSE Evolution During Training', fontsize=18, fontweight='bold', pad=20)

# 优化网格和刻度
ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
ax.tick_params(axis='both', which='major', labelsize=14, width=1.2, length=6)

# 设置x轴范围和刻度
ax.set_xlim(min(x_values), max(x_values))
ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # 限制x轴刻度数量

# 添加图例（放在右上角外侧）
legend = ax.legend(loc='upper right', 
                    bbox_to_anchor=(1.0, 1.0),
                    fontsize=14,
                    frameon=False,
                    title='Mask Strategies',
                    title_fontsize=15,
                    handletextpad=0.5,
                    columnspacing=1.0)

# 美化边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 保存高分辨率图像
output_path = Path("results") / f"nmse_evolution.png"

pdf_path = output_path.with_suffix('.pdf')
fig.savefig(pdf_path)
print(f"PDF版本已保存至: {pdf_path}")
