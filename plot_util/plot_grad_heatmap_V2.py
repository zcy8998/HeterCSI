import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# 配置绘图风格
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

def load_batch_data(data_dir, epoch, step):
    """
    读取指定 step 的所有样本的梯度向量
    """
    data_dir = Path(data_dir)
    # 查找该 step 下的所有梯度文件
    grad_files = sorted(glob.glob(str(data_dir / f"grad_epoch{epoch:04d}_step{step:06d}_sample*.npy")))

    if not grad_files:
        print(f"[Warning] No gradient data found for Epoch {epoch} Step {step} in {data_dir}")
        return None

    grads = []
    for g_file in grad_files:
        g = np.load(g_file)
        grads.append(g)

    return np.stack(grads)

def plot_gradient_analysis(schemes, epoch, step, save_path=None):
    """
    对比不同训练方案在同一个 Batch 内的梯度余弦相似度
    """
    num_schemes = len(schemes)
    if num_schemes == 0:
        return

    # 创建画布：1 行 N 列
    fig, axes = plt.subplots(1, num_schemes, figsize=(6 * num_schemes, 7))
    
    if num_schemes == 1:
        axes = [axes]
    
    for idx, (scheme_name, data_dir) in enumerate(schemes.items()):
        print(f"Processing {scheme_name} from {data_dir}...")
        grads = load_batch_data(data_dir, epoch, step)
        
        ax = axes[idx]

        if grads is None:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        batch_size = grads.shape[0]
        
        # 1. 计算余弦相似度矩阵
        # 假设梯度向量已经是归一化的，或者在这里不做归一化直接看方向一致性（通常建议归一化）
        # 这里为了计算严谨的余弦相似度，手动归一化一下
        norms = np.linalg.norm(grads, axis=1, keepdims=True)
        # 防止除零
        norms[norms == 0] = 1e-8
        normalized_grads = grads / norms
        
        sim_matrix = np.dot(normalized_grads, normalized_grads.T)
        
        # 绘制热力图
        im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1) 
        
        # ax.set_title(f"[{scheme_name}]\nCosine Similarity Matrix", fontsize=14, fontweight='bold')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        
        # 2. 统计负值比例 (Negative Ratio)
        mask = ~np.eye(batch_size, dtype=bool) # 排除对角线
        
        if batch_size > 1:
            off_diag_sims = sim_matrix[mask]
            
            # 统计负值 (Conflict)
            neg_count = np.sum(off_diag_sims < 0)
            total_pairs = len(off_diag_sims)
            neg_ratio = neg_count / total_pairs
            
            # 统计正值 (Alignment) - 可选
            # pos_ratio = 1.0 - neg_ratio
            
            # 平均相似度 (越高越好)
            avg_sim = np.mean(off_diag_sims)
        else:
            neg_ratio = 0.0
            avg_sim = 1.0
            
        # 3. 在图下方标注统计信息
        # 重点显示负值比例
        stats_text = (f"Conflict Ratio: {neg_ratio:.2%}\n")
        print(stats_text)
        
        # 绿底框表示“好”（如果负值低），红底框表示“差”？这里统一用灰白框即可
        # ax.text(0.5, -0.22, stats_text, transform=ax.transAxes, 
        #         ha='center', va='top', fontsize=13, fontweight='medium',
        #         bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="black", alpha=0.9))
        
        # 仅在最后一个子图显示 Colorbar
        if idx == num_schemes - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Cosine Similarity")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22) # 留出底部空间给文字
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

# --- 使用示例 ---
if __name__ == "__main__":
    base_results_dir = Path("./results")
    
    comparison_schemes = {
        # "Global Strategy": base_results_dir / "global",
        "Bucket Strategy": base_results_dir / "bucket"
    }

    target_epoch = 0
    target_step = 356

    output_filename = f"results/comparison_E{target_epoch}_S{target_step}_ratio.pdf"

    plot_gradient_analysis(
        schemes=comparison_schemes,
        epoch=target_epoch,
        step=target_step,
        save_path=output_filename
    )