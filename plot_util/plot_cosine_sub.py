# save as plot_grad_cosines_pdf.py
# -*- coding: utf-8 -*-
import os
import csv
from pathlib import Path
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shutil
import gc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, leave=True):
        print(f"Processing: {desc}")
        return iterable
    
# 获取缓存路径
cache_dir = matplotlib.get_cachedir()

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
    'grid.alpha': 0.6
})

# ================= 用户配置区域 (User Config) =================
RESULTS_ROOT = "results"     # 根目录
OUTPUT_DIR = "results"       # 图片输出目录

# 定义需要生成的图片配置
# 配置格式: (数据集A, 任务A, 数据集B, 任务B, 子图标题)
FIGURE_CONFIGS = [
    {
        "filename": "D1_Comparison_Triple",  # 输出文件名 (不带后缀)
        "subplots": [
            # 1. Top Plot
            ("D1", "temporal_16batch", "D40", "temporal_16batch", "Top: D1 vs D40"),
            
            # # 2. Middle Plot
            # ("D1", "temporal_16batch", "D100", "temporal_16", "Mid: D1 vs D100"),

            # 3. Bottom Plot
            ("D1", "temporal_16batch", "D5", "temporal_16", "Bot: D1 vs D5")
        ]
    },
]

# 参数设置
MAX_PER_EPOCH = 500             
MAX_SAMPLES_FOR_HIST = 50000    
GLOB_PATTERN = "grad_*.npy"
BINS = 150                      
BLOCK_SIZE = 1024               
RANDOM_SEED = 42

# 绘图颜色
NEG_COLOR = "#D55E00"           
POS_COLOR = "#0072B2"           
# ============================================================

np.random.seed(RANDOM_SEED)
EPOCH_RE = re.compile(r"epoch[_\-]?0*([0-9]+)", re.IGNORECASE)

def parse_epoch_from_filename(fname: str):
    m = EPOCH_RE.search(fname)
    return int(m.group(1)) if m else None

def get_grouped_files(base_root, dataset_name, task_name, pattern):
    target_dir = Path(base_root) / dataset_name / task_name
    if not target_dir.exists():
        print(f"[Warning] Directory not found: {target_dir}")
        return {}
    
    files = sorted(target_dir.glob(pattern))
    groups = {}
    for f in files:
        ep = parse_epoch_from_filename(f.name)
        if ep is None: continue
        groups.setdefault(ep, []).append(f)
    
    final_groups = {}
    for ep, flist in groups.items():
        flist = sorted(flist)
        if MAX_PER_EPOCH and len(flist) > MAX_PER_EPOCH:
            flist = flist[:MAX_PER_EPOCH]
        final_groups[ep] = flist
    return final_groups

def load_matrix(flist):
    if not flist:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = [np.load(str(f)).astype(np.float32).ravel() for f in flist]
    mat = np.stack(vecs, axis=0)
    # L2 Normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms

def compute_stats_and_sample(matA, matB, current_samples, max_samples):
    nA = matA.shape[0]
    s_sum = 0.0
    s2_sum = 0.0
    cnt = 0
    neg_cnt = 0
    new_samples = []
    need_sample = len(current_samples) < max_samples
    
    for i in range(0, nA, BLOCK_SIZE):
        end = min(i + BLOCK_SIZE, nA)
        blockA = matA[i:end]
        sim_block = blockA @ matB.T
        
        s_sum += float(np.sum(sim_block))
        s2_sum += float(np.sum(sim_block ** 2))
        neg_cnt += int(np.sum(sim_block < 0))
        cnt += sim_block.size
        
        if need_sample:
            flat_sims = sim_block.ravel()
            remaining_quota = max_samples - len(current_samples) - len(new_samples)
            if remaining_quota > 0:
                if len(flat_sims) <= remaining_quota:
                    new_samples.extend(flat_sims.tolist())
                else:
                    chosen = np.random.choice(flat_sims, size=remaining_quota, replace=False)
                    new_samples.extend(chosen.tolist())
                    need_sample = False
    return s_sum, s2_sum, cnt, neg_cnt, new_samples

# ================= CSV 读写功能 =================

def save_csv(path, stats, samples):
    mean_val, std_val, neg_frac = stats
    print(f"  -> Saving data to CSV: {path}")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["__METADATA__", "Value"])
        writer.writerow(["global_mean", mean_val])
        writer.writerow(["global_std", std_val])
        writer.writerow(["global_neg_frac", neg_frac])
        writer.writerow([])
        writer.writerow(["sampled_cosines"])
        for val in samples:
            writer.writerow([f"{val:.6f}"])

def load_csv(path):
    print(f"  -> Loading data from CSV: {path}")
    stats_dict = {}
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        mode = "meta"
        for row in reader:
            if not row: continue
            if row[0] == "sampled_cosines":
                mode = "data"
                continue
            if mode == "meta":
                if row[0] == "__METADATA__": continue
                if len(row) >= 2:
                    try:
                        stats_dict[row[0]] = float(row[1])
                    except ValueError: pass
            elif mode == "data":
                try:
                    samples.append(float(row[0]))
                except ValueError: pass
    stats = (
        stats_dict.get("global_mean", 0.0),
        stats_dict.get("global_std", 0.0),
        stats_dict.get("global_neg_frac", 0.0)
    )
    return stats, samples

# ================= 核心逻辑封装 =================

def get_data_for_comparison(d1, t1, d2, t2, out_dir):
    base_name = f"{d1}_{t1}__vs__{d2}_{t2}"
    csv_path = out_dir / f"{base_name}.csv"
    
    # 1. 尝试加载缓存
    if csv_path.exists():
        print(f"  [Cache Hit] {base_name}")
        return load_csv(csv_path)
    
    # 2. 计算
    print(f"  [Computing] {base_name} ...")
    files_map_A = get_grouped_files(RESULTS_ROOT, d1, t1, GLOB_PATTERN)
    files_map_B = get_grouped_files(RESULTS_ROOT, d2, t2, GLOB_PATTERN)
    
    if not files_map_A or not files_map_B:
        print("  -> Skipping: Directory missing or empty.")
        return (0,0,0), []
        
    common_epochs = sorted(set(files_map_A.keys()) & set(files_map_B.keys()))
    if not common_epochs:
        print("  -> No common epochs found.")
        return (0,0,0), []
    
    total_sum = 0.0
    total_sum2 = 0.0
    total_count = 0
    total_neg = 0
    sampled_cosines = []
    
    pbar = tqdm(common_epochs, desc="  Epochs", leave=False)
    for ep in pbar:
        fA = files_map_A[ep]
        fB = files_map_B[ep]
        matA = load_matrix(fA)
        matB = load_matrix(fB)
        
        s, s2, cnt, neg, samples = compute_stats_and_sample(
            matA, matB, sampled_cosines, MAX_SAMPLES_FOR_HIST
        )
        
        total_sum += s
        total_sum2 += s2
        total_count += cnt
        total_neg += neg
        sampled_cosines.extend(samples)
        
        del matA, matB, samples
        gc.collect()

    if total_count > 0:
        mean_cos = total_sum / total_count
        var_cos = (total_sum2 / total_count) - (mean_cos ** 2)
        std_cos = np.sqrt(max(0.0, var_cos))
        neg_frac = total_neg / total_count
    else:
        mean_cos = std_cos = neg_frac = 0.0
    
    stats = (mean_cos, std_cos, neg_frac)
    save_csv(csv_path, stats, sampled_cosines)
    return stats, sampled_cosines

# ================= 绘图 =================

def plot_multi_histograms(subplot_data_list, out_path):
    """
    绘制多个子图（上下分布）的 PDF。
    支持任意数量子图，自动调整高度。
    """
    num_plots = len(subplot_data_list)
    if num_plots == 0:
        return

    # 动态调整高度：每个子图分配约 3.5 英寸高度，保持“扁平”
    fig_height = 3.5 * num_plots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, fig_height), sharex=False)
    
    # 如果只有1个子图，matplotlib返回的是Axes对象而不是列表，需转换
    if num_plots == 1:
        axes = [axes]

    for idx, (ax, item) in enumerate(zip(axes, subplot_data_list)):
        data = item['data']
        stats = item['stats']
        title = item['title']
        
        if not data:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        arr = np.array(data)
        counts, bin_edges = np.histogram(arr, bins=BINS)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        width = bin_edges[1] - bin_edges[0]
        
        colors = [NEG_COLOR if c < 0 else POS_COLOR for c in bin_centers]
        
        ax.bar(bin_centers, counts, width=width * 0.95, color=colors, edgecolor='none')
        
        # 装饰
        ax.set_ylabel("Frequency")
        # ax.set_title(title, pad=10, fontweight='bold') # 如果想要标题可以取消注释
        ax.axvline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 统计信息框
        mean_val, std_val, neg_frac = stats
        stats_txt = (f"Neg%: {neg_frac:.2%}")
        ax.text(0.98, 0.95, stats_txt, transform=ax.transAxes, 
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cccccc"))
        
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        # 设置X轴标签
        ax.set_xlabel("Cosine Similarity")
    
    # 调整布局，防止重叠
    plt.tight_layout()
    print(f"  -> Saving multi-plot to {out_path}")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path, format='eps', bbox_inches='tight')
    plt.close()

# ================= Main =================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for fig_conf in FIGURE_CONFIGS:
        fname = fig_conf["filename"]
        subplots_conf = fig_conf["subplots"] 
        
        print(f"\nGenerating Figure: {fname}")
        
        plot_data_list = []
        
        # 收集每个子图的数据
        for (d1, t1, d2, t2, title_text) in subplots_conf:
            print(f"  Processing Subplot: {title_text} ({d1}/{t1} vs {d2}/{t2})")
            stats, samples = get_data_for_comparison(d1, t1, d2, t2, out_dir)
            plot_data_list.append({
                "data": samples,
                "stats": stats,
                "title": title_text
            })
            
        # 只要有数据就画图
        if len(plot_data_list) > 0:
            pdf_path = out_dir / f"{fname}.pdf"
            plot_multi_histograms(plot_data_list, pdf_path)
        else:
            print(f"  [Skip] No data found for {fname}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
