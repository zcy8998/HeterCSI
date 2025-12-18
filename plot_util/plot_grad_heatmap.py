import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os


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

def load_batch_data(data_dir, epoch, step):
    """
    读取指定 step 的所有样本的梯度向量和幅值
    """
    data_dir = Path(data_dir)
    # 查找该 step 下的所有样本文件
    # 假设文件名格式为: ...sample0000.npy, ...sample0001.npy (0填充确保排序正确)
    grad_files = sorted(glob.glob(str(data_dir / f"grad_epoch{epoch:04d}_step{step:06d}_sample*.npy")))
    mag_files = sorted(glob.glob(str(data_dir / f"mag_epoch{epoch:04d}_step{step:06d}_sample*.npy")))

    if not grad_files:
        print(f"[Warning] No data found for Epoch {epoch} Step {step} in {data_dir}")
        return None, None
    
    if len(grad_files) != len(mag_files):
        print(f"[Error] Mismatch in number of gradient files ({len(grad_files)}) and magnitude files ({len(mag_files)})")
        return None, None

    grads = []
    mags = []

    # 读取文件
    for g_file, m_file in zip(grad_files, mag_files):
        # 简单校验文件名后缀是否对应，防止错位
        g_suffix = g_file.split('_')[-1] # sampleXXXX.npy
        m_suffix = m_file.split('_')[-1]
        if g_suffix != m_suffix:
            print(f"[Error] File mismatch: {g_file} vs {m_file}")
            continue

        g = np.load(g_file)
        m = np.load(m_file)
        grads.append(g)
        mags.append(float(m))

    # grads shape: (Batch_Size, Vector_Dim)
    # mags shape: (Batch_Size,)
    return np.stack(grads), np.array(mags)

def plot_gradient_analysis(schemes, epoch, step, save_path=None):
    """
    对比不同训练方案在同一个 Batch 内的梯度特性
    
    Args:
        schemes (dict): 字典，格式为 {'方案名': '保存梯度的文件夹路径'}
        epoch (int): 要分析的 Epoch
        step (int): 要分析的 Step
    """
    num_schemes = len(schemes)
    if num_schemes == 0:
        return

    # 创建画布：
    # Row 0: 相似度矩阵 (Heatmap)
    # Row 1: 幅值柱状图 (Bar Chart)
    fig, axes = plt.subplots(2, num_schemes, figsize=(6 * num_schemes, 12))
    
    # 维度处理，确保 axes 始终是二维数组 [row, col]
    if num_schemes == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    # 遍历不同的方案 (Global, Bucket)
    for idx, (scheme_name, data_dir) in enumerate(schemes.items()):
        print(f"Processing {scheme_name} from {data_dir}...")
        grads, mags = load_batch_data(data_dir, epoch, step)
        
        if grads is None:
            # 如果没读到数据，把图清空显示文字
            axes[0, idx].text(0.5, 0.5, "No Data", ha='center')
            axes[1, idx].text(0.5, 0.5, "No Data", ha='center')
            continue

        batch_size = grads.shape[0]
        
        # ---------------------------------------------------------
        # 1. 绘制余弦相似度矩阵 (Intra-Batch Cosine Similarity)
        # ---------------------------------------------------------
        # 计算两两之间的点积 (因为已经归一化，点积=余弦相似度)
        sim_matrix = np.dot(grads, grads.T)
        
        ax_sim = axes[0, idx]
        # vmin=-1, vmax=1 确保颜色映射在理论范围内，方便不同方案横向对比
        im = ax_sim.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1) 
        
        ax_sim.set_title(f"[{scheme_name}]\nCosine Similarity Matrix", fontsize=14, fontweight='bold')
        ax_sim.set_xlabel("Sample Index")
        ax_sim.set_ylabel("Sample Index")
        
        # 计算统计量：平均相似度 (排除对角线上的1.0)
        mask = ~np.eye(batch_size, dtype=bool)
        if batch_size > 1:
            avg_sim = sim_matrix[mask].mean()
            std_sim = sim_matrix[mask].std()
        else:
            avg_sim, std_sim = 1.0, 0.0
            
        # 在图下方标注统计信息
        stats_text = (f"Avg Pairwise Sim: {avg_sim:.4f}\n"
                      f"Sim Std Dev: {std_sim:.4f}")
        ax_sim.text(0.5, -0.2, stats_text, transform=ax_sim.transAxes, 
                    ha='center', va='top', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        
        # 仅在最后一个子图显示 Colorbar，避免拥挤
        if idx == num_schemes - 1:
            cbar = fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)
            cbar.set_label("Cosine Similarity")

        # ---------------------------------------------------------
        # 2. 绘制梯度幅值 (Gradient Magnitudes)
        # ---------------------------------------------------------
        ax_mag = axes[1, idx]
        indices = np.arange(batch_size)
        
        # 绘制柱状图
        bars = ax_mag.bar(indices, mags, color='steelblue', edgecolor='black', alpha=0.8)
        
        ax_mag.set_title(f"[{scheme_name}]\nGradient Magnitudes (L2 Norm)", fontsize=14, fontweight='bold')
        ax_mag.set_xlabel("Sample Index")
        ax_mag.set_ylabel("Magnitude")
        ax_mag.grid(axis='y', linestyle='--', alpha=0.4)
        
        # 计算统计量
        mean_mag = np.mean(mags)
        std_mag = np.std(mags)
        max_mag = np.max(mags)
        
        # 绘制均值线
        ax_mag.axhline(mean_mag, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_mag:.2f}')
        ax_mag.legend(loc='upper right')

        # 在图下方标注统计信息
        mag_stats_text = (f"Mean Mag: {mean_mag:.4f}\n"
                          f"Std Dev: {std_mag:.4f}\n"
                          f"Max Mag: {max_mag:.4f}")
        ax_mag.text(0.5, -0.25, mag_stats_text, transform=ax_mag.transAxes, 
                    ha='center', va='top', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    # 调整底部留白以显示统计文字
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置：你要对比的方案名称和对应的文件夹路径
    # 假设你的根目录是 ./results
    
    base_results_dir = Path("./results")
    
    comparison_schemes = {
        "Global Strategy": base_results_dir / "global",
        "Bucket Strategy": base_results_dir / "bucket"
    }

    # 指定要查看的 Epoch 和 Step
    target_epoch = 1
    target_step = 1544  # 确保这个 step 存在于两个文件夹中

    # 输出图片路径
    output_filename = f"results/comparison_E{target_epoch}_S{target_step}.pdf"

    plot_gradient_analysis(
        schemes=comparison_schemes,
        epoch=target_epoch,
        step=target_step,
        save_path=output_filename
    )