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
print(f"Matplotlib 缓存目录位于: {cache_dir}")

# 删除缓存目录
# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)
#     print("缓存已清除，请重启你的 Python kernel/脚本！")
# else:
#     print("未找到缓存目录。")

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,  # 增大基础字号
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
OUTPUT_DIR = "results"     # 图片输出目录

# 定义需要对比的配对：(数据集A, 任务A, 数据集B, 任务B)
COMPARISON_CONFIG = [
    ("D1", "temporal_16batch", "D40", "temporal_16batch"),
    # ("D1", "temporal", "D1", "freq"), 
]

# 参数设置
MAX_PER_EPOCH = 500             # 每个Epoch最多读取多少个梯度文件
MAX_SAMPLES_FOR_HIST = 50000    # 用于画直方图的采样点上限
GLOB_PATTERN = "grad_*.npy"
BINS = 150                      # 直方图柱子数量
BLOCK_SIZE = 1024               # 计算时的显存/内存分块大小
RANDOM_SEED = 42

# 绘图颜色
NEG_COLOR = "#D55E00"           # 橙色 (负相关)
POS_COLOR = "#0072B2"           # 蓝色 (正相关)
# ============================================================

np.random.seed(RANDOM_SEED)

EPOCH_RE = re.compile(r"epoch[_\-]?0*([0-9]+)", re.IGNORECASE)

def parse_epoch_from_filename(fname: str):
    """从文件名解析 Epoch 编号"""
    m = EPOCH_RE.search(fname)
    return int(m.group(1)) if m else None

def get_grouped_files(base_root, dataset_name, task_name, pattern):
    """获取指定数据集和任务路径下的文件，按 Epoch 分组"""
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
    
    # 限制每个 Epoch 的文件数量并排序
    final_groups = {}
    for ep, flist in groups.items():
        flist = sorted(flist)
        if MAX_PER_EPOCH and len(flist) > MAX_PER_EPOCH:
            flist = flist[:MAX_PER_EPOCH]
        final_groups[ep] = flist
    return final_groups

def load_matrix(flist):
    """加载文件列表为标准化后的矩阵 (N, D)"""
    if not flist:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = [np.load(str(f)).astype(np.float32).ravel() for f in flist]
    mat = np.stack(vecs, axis=0)
    
    # L2 Normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms

def compute_stats_and_sample(matA, matB, current_samples, max_samples):
    """计算统计量（全量）并进行采样"""
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
    """
    保存 CSV 文件
    Format:
      row 1-3: metadata (mean, std, neg_frac)
      row 4: header for samples
      row 5+: sample values
    """
    mean_val, std_val, neg_frac = stats
    print(f"  -> Saving data to CSV: {path}")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入全局统计信息
        writer.writerow(["__METADATA__", "Value"])
        writer.writerow(["global_mean", mean_val])
        writer.writerow(["global_std", std_val])
        writer.writerow(["global_neg_frac", neg_frac])
        
        # 写入采样数据
        writer.writerow([]) # 空行分隔
        writer.writerow(["sampled_cosines"])
        for val in samples:
            writer.writerow([f"{val:.6f}"])

def load_csv(path):
    """读取 CSV 文件，返回 (stats, samples)"""
    print(f"  -> Loading data from CSV: {path}")
    stats_dict = {}
    samples = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        mode = "meta" # meta or data
        
        for row in reader:
            if not row: continue # skip empty lines
            
            if row[0] == "sampled_cosines":
                mode = "data"
                continue
            
            if mode == "meta":
                if row[0] == "__METADATA__": continue
                # 读取 metadata
                if len(row) >= 2:
                    try:
                        stats_dict[row[0]] = float(row[1])
                    except ValueError:
                        pass
            elif mode == "data":
                try:
                    samples.append(float(row[0]))
                except ValueError:
                    pass
                    
    # 组装返回
    stats = (
        stats_dict.get("global_mean", 0.0),
        stats_dict.get("global_std", 0.0),
        stats_dict.get("global_neg_frac", 0.0)
    )
    return stats, samples

# ===============================================

def plot_histogram(data, title, out_path, stats):
    """绘制并保存直方图 (PDF)"""
    if not data:
        print("  [Error] No data to plot.")
        return


    arr = np.array(data)
    d_min, d_max = arr.min(), arr.max()
    
    counts, bin_edges = np.histogram(arr, bins=BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    width = bin_edges[1] - bin_edges[0]
    
    colors = [NEG_COLOR if c < 0 else POS_COLOR for c in bin_centers]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bin_centers, counts, width=width * 0.95, color=colors, edgecolor='none')

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    # ax.set_title(title)
    
    ax.axvline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

    mean_val, std_val, neg_frac = stats
    # stats_txt = (f"Mean: {mean_val:.4f}\n"
    #              f"Std:  {std_val:.4f}\n"
    #              f"Neg%: {neg_frac:.2%}\n"
    #              f"Range: [{d_min:.3f}, {d_max:.3f}]")
    
    stats_txt = (f"Neg%: {neg_frac:.2%}")
    
    ax.text(0.98, 0.95, stats_txt, transform=ax.transAxes, 
            ha='right', va='top', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cccccc"))

    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    print(f"  -> Saving plot to {out_path}")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for (d1, t1, d2, t2) in COMPARISON_CONFIG:
        print(f"\nTask: [{d1}/{t1}] vs [{d2}/{t2}]")
        
        # 定义文件名
        base_name = f"{d1}_{t1}__vs__{d2}_{t2}"
        csv_path = out_dir / f"{base_name}.csv"
        pdf_path = out_dir / f"{base_name}.pdf"
        title = f"{d1} ({t1}) vs {d2} ({t2})"

        # 1. 检查是否存在 CSV 缓存
        if csv_path.exists():
            # HIT CACHE
            print("  -> Found cached CSV. Skipping computation.")
            final_stats, sampled_cosines = load_csv(csv_path)
        
        else:
            # MISS CACHE - COMPUTE
            files_map_A = get_grouped_files(RESULTS_ROOT, d1, t1, GLOB_PATTERN)
            files_map_B = get_grouped_files(RESULTS_ROOT, d2, t2, GLOB_PATTERN)
            
            if not files_map_A or not files_map_B:
                print("  -> Skipping: One of the directories is empty or missing.")
                continue
                
            common_epochs = sorted(set(files_map_A.keys()) & set(files_map_B.keys()))
            if not common_epochs:
                print("  -> No common epochs found.")
                continue
            
            print(f"  -> Found {len(common_epochs)} common epochs. Computing...")

            total_sum = 0.0
            total_sum2 = 0.0
            total_count = 0
            total_neg = 0
            sampled_cosines = []
            
            pbar = tqdm(common_epochs, desc="Computing", leave=False)
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
            
            final_stats = (mean_cos, std_cos, neg_frac)
            print(f"  -> Computed: Mean={mean_cos:.4f}, Std={std_cos:.4f}, Neg={neg_frac:.2%}")

            # 保存结果到 CSV
            save_csv(csv_path, final_stats, sampled_cosines)

        # 2. 统一绘图
        plot_histogram(
            sampled_cosines, 
            title, 
            pdf_path, 
            final_stats
        )

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()