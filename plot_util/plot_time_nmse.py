import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker
import numpy as np

# --------------------------
# 1. 配置区域 (Configuration)
# --------------------------

# 限制最大时间 (单位: 分钟)
# 设置为 具体数字 (例如 2000) 则截断超过该时间的数据
# 设置为 None 则绘制所有数据
MAX_TIME_MINUTES = 1000 

SAVE_DIR = Path("results")

# 实验列表配置
experiments_config = [
    {
        "name": "Global",
        "path": "/data/zcy_new/cross_csi/global_24dataset_256batch/nmse_results_all_global.csv",
        "duration": 930,       # 930秒/epoch
        "filter_col": "mask_type",
        "filter_val": "freq"
    },
    {
        "name": "Proposed",
        "path": "/data/zcy_new/cross_csi/bucket_24dataset_256batch/nmse_results_all_bucket.csv",
        "duration": 810,       # 810秒/epoch
        "filter_col": "mask_type",
        "filter_val": "freq"
    },
    {   
        "name": "Alternating",
        "path": "/data/zcy_new/cross_csi/seq_24dataset_256batch/nmse_results_all_seq.csv",
        "duration": 810,
        "filter_col": "mask_type",
        "filter_val": "freq"
    }
]

SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
    '#1f77b4', '#d62728', '#2ca02c', 
    '#ff7f0e', '#9467bd', '#8c564b', '#e377c2'
]

# --------------------------
# 3. 数据读取与处理
# --------------------------

plot_data = [] 

print(f"开始读取 CSV 数据 (最大时间限制: {MAX_TIME_MINUTES if MAX_TIME_MINUTES else '无'} 分钟)...")

for config in experiments_config:
    exp_name = config["name"]
    file_path = Path(config["path"])
    duration = config.get("duration", 1.0)
    f_col = config.get("filter_col")
    f_val = config.get("filter_val")
    
    print(f"正在处理: {exp_name}")
    
    if not file_path.exists():
        print(f"  错误: 文件不存在 {file_path}，跳过。")
        continue
        
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        if 'epoch' not in df.columns or 'avg_nmse' not in df.columns:
            print(f"  警告: {exp_name} 缺少必要列，跳过。")
            continue

        # 筛选逻辑 (mask_type 等)
        if f_col is not None and f_val is not None:
            if f_col in df.columns:
                df = df[df[f_col] == f_val]
                if df.empty: continue
            else:
                continue

        df['avg_nmse'] = pd.to_numeric(df['avg_nmse'], errors='coerce')
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df = df.dropna(subset=['epoch', 'avg_nmse'])
        
        # 按 epoch 聚合
        epoch_avg = df.groupby('epoch')['avg_nmse'].mean().sort_index()
        
        epochs = epoch_avg.index.to_numpy()
        nmse_values = epoch_avg.values
        
        if len(epochs) == 0: continue

        # 1. 先计算所有点的时间 (分钟)
        time_values = (epochs * duration) / 60.0 
        
        # 2. 根据时间进行截断
        if MAX_TIME_MINUTES is not None:
            mask = time_values <= MAX_TIME_MINUTES
            time_values = time_values[mask]
            nmse_values = nmse_values[mask]
            
            if len(time_values) == 0:
                print(f"  警告: 数据越界，跳过。")
                continue
            
            print(f"  - 时间截断: 保留前 {MAX_TIME_MINUTES} 分钟 (剩余数据点: {len(time_values)})")
        else:
            print(f"  - 保留所有数据 (最大: {time_values.max():.2f} min)")

        plot_data.append({
            "name": exp_name,
            "x": time_values,
            "y": nmse_values
        })
                
    except Exception as e:
        print(f"  读取错误 {file_path}: {e}")

# --------------------------
# 4. 绘图
# --------------------------

if not plot_data:
    raise ValueError("没有读取到任何有效数据。")

fig, ax = plt.subplots(figsize=(8, 6.5))

all_times_flat = [] 

for i, item in enumerate(plot_data):
    color = COLORS[i % len(COLORS)]
    x_vals = item["x"]
    y_vals = item["y"]
    
    all_times_flat.extend(x_vals)
    
    ax.plot(x_vals, y_vals, 
            color=color, 
            linewidth=2.5,
            label=item["name"],
            zorder=10)

ax.set_xlabel('Training Time (minutes)', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('NMSE', fontsize=16, fontweight='bold', labelpad=10)

# 【修改】禁用科学计数法，强制使用普通数字显示
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False) # 禁用科学计数法
formatter.set_useOffset(False)  # 禁用偏移量（防止出现 +1e3 这种）
ax.xaxis.set_major_formatter(formatter)

ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
ax.tick_params(axis='both', which='major', labelsize=14, width=1.2, length=6)

# 动态设置 X 轴范围
if MAX_TIME_MINUTES:
    ax.set_xlim(0, MAX_TIME_MINUTES) 
elif all_times_flat:
    ax.set_xlim(0, max(all_times_flat) * 1.05)

legend = ax.legend(loc='upper right', 
                   fontsize=13,
                   frameon=False,
                   title_fontsize=15)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# --------------------------
# 5. 保存
# --------------------------
output_filename = "nmse_vs_time_plain"
pdf_path = SAVE_DIR / f"{output_filename}.pdf"
png_path = SAVE_DIR / f"{output_filename}.png"

fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"\n绘图完成！")
print(f"PDF已保存至: {pdf_path}")
print(f"PNG已保存至: {png_path}")
