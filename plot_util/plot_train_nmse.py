import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker

# --------------------------
# 1. 配置区域 (请修改这里)
# --------------------------

experiment_paths = {
    "Global": "/data/zcy_new/cross_csi/global_24dataset_256batch/nmse_results_all_global.csv",
    "Proposed": "/data/zcy_new/cross_csi/bucket_24dataset_256batch/nmse_results_all_bucket.csv",
    "Alternating": "/data/zcy_new/cross_csi/seq_24dataset_256batch/nmse_results_all_seq.csv"
}

save_dir = Path("results")
save_dir.mkdir(parents=True, exist_ok=True)

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
    '#ff7f0e', '#9467bd', '#8c564b'
]

# --------------------------
# 3. 数据读取 (增加健壮性处理)
# --------------------------

all_data = {}

print("开始读取 CSV 数据...")

for exp_name, file_path_str in experiment_paths.items():
    file_path = Path(file_path_str)
    print(f"正在处理: {exp_name} -> {file_path}")
    
    if not file_path.exists():
        print(f"  错误: 文件不存在 {file_path}，跳过。")
        continue
        
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 清理列名空格
        df.columns = df.columns.str.strip()
        
        # 检查关键列是否存在
        if 'epoch' not in df.columns or 'avg_nmse' not in df.columns:
            print(f"  警告: 列名不匹配，现有列名: {df.columns}")
            continue

        # =======================================================
        # 关键修复步骤：强制类型转换
        # =======================================================
        
        # 1. 将 'avg_nmse' 列转换为数字，无法转换的变为 NaN (例如文件中间夹杂的表头)
        df['avg_nmse'] = pd.to_numeric(df['avg_nmse'], errors='coerce')
        
        # 2. 将 'epoch' 列也转换为数字
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        
        # 3. 删除包含 NaN 的行 (剔除脏数据)
        original_len = len(df)
        df = df.dropna(subset=['epoch', 'avg_nmse'])
        dropped_len = original_len - len(df)
        if dropped_len > 0:
            print(f"  - 已剔除 {dropped_len} 行脏数据 (非数字内容)")

        # =======================================================
        
        # 核心计算: 按 epoch 分组求均值
        epoch_avg = df.groupby('epoch')['avg_nmse'].mean()
        
        # 排序并提取
        epoch_avg = epoch_avg.sort_index()
        x_values = epoch_avg.index.to_numpy()
        y_values = epoch_avg.values
        
        if len(x_values) == 0:
            print(f"  警告: {exp_name} 数据为空")
            continue
            
        all_data[exp_name] = (x_values, y_values)
        print(f"  - 成功读取 {len(x_values)} 个 Epoch 数据")
                
    except Exception as e:
        print(f"  读取错误 {file_path}: {e}")

# --------------------------
# 4. 绘图
# --------------------------

if not all_data:
    raise ValueError("没有读取到任何有效数据，请检查路径。")

fig, ax = plt.subplots(figsize=(8, 7))

all_epochs_flat = [] 

for i, (exp_name, (x_values, y_values)) in enumerate(all_data.items()):
    
    color = COLORS[i % len(COLORS)]
    all_epochs_flat.extend(x_values)
    
    ax.plot(x_values, y_values, 
            color=color, 
            linewidth=2.5,
            linestyle='-',        
            markersize=6,
            markeredgewidth=1.5,
            markeredgecolor='white', 
            markerfacecolor=color,
            label=exp_name,
            zorder=10)

ax.set_xlabel('Training Epochs', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('Average NMSE', fontsize=16, fontweight='bold', labelpad=10)

ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
ax.tick_params(axis='both', which='major', labelsize=14, width=1.2, length=6)

if all_epochs_flat:
    ax.set_xlim(min(all_epochs_flat), max(all_epochs_flat))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

legend = ax.legend(loc='upper right', 
                   fontsize=14,
                   frameon=False,
                   title='Experiments',
                   title_fontsize=15)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# --------------------------
# 5. 保存
# --------------------------
output_filename = "csv_nmse_comparison_plot"
pdf_path = save_dir / f"{output_filename}.pdf"
png_path = save_dir / f"{output_filename}.png"

fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"\n绘图完成！")
print(f"PDF已保存至: {pdf_path}")
print(f"PNG已保存至: {png_path}")

plt.show()