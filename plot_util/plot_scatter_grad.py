import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.lines import Line2D

def plot_single_pair(path_a, path_b, label_a, label_b, output_dir, file_name, max_points=100000):
    """
    绘制单张对比图并保存为PDF
    """
    # --- 1. 读取数据 ---
    try:
        grad_a = np.load(path_a)
        grad_b = np.load(path_b)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # --- 2. 计算统计指标 ---
    norm_a = np.linalg.norm(grad_a)
    norm_b = np.linalg.norm(grad_b)
    # 防止除零
    denom = norm_a * norm_b
    if denom == 0: denom = 1e-8
    
    cos_sim = np.dot(grad_a, grad_b) / denom
    
    conflict_mask_all = (grad_a * grad_b) < 0
    conflict_ratio = np.sum(conflict_mask_all) / len(grad_a)
    
    # --- 3. 下采样 (为了绘图速度和文件大小) ---
    total_points = len(grad_a)
    if total_points > max_points:
        indices = np.random.choice(total_points, max_points, replace=False)
        x = grad_a[indices]
        y = grad_b[indices]
    else:
        x = grad_a
        y = grad_b

    # --- 4. 绘图设置 ---
    # 颜色: 冲突为红色，一致为绿色
    mask_conflict = (x * y) < 0
    colors = np.where(mask_conflict, '#CD5C5C', '#2E8B57')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8)) # 稍微调整尺寸适应PDF

    ax.scatter(x, y, c=colors, s=10, alpha=0.5, edgecolors='none', rasterized=True) 
    # 注意: rasterized=True 对于大量散点保存PDF非常重要，否则PDF会巨大且渲染极慢

    # --- 5. 辅助线 ---
    limit = max(np.abs(x.max()), np.abs(x.min()), np.abs(y.max()), np.abs(y.min())) * 1.1

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.plot([-limit, limit], [-limit, limit], color='green', linestyle='--', alpha=0.5, label='Alignment')
    ax.plot([-limit, limit], [limit, -limit], color='red', linestyle=':', alpha=0.5, label='Conflict')

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # 提取 Step 信息用于标题 (假设文件名格式: grad_epoch0000_step000041.npy)
    step_info = file_name.replace('.npy', '').replace('grad_', '')
    
    ax.set_xlabel(f"Gradient: {label_a}", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"Gradient: {label_b}", fontsize=12, fontweight='bold')
    ax.set_title(f"Gradient Analysis: {label_a} vs {label_b}\n({step_info})", fontsize=13, pad=10)

    # --- 6. 统计信息框 ---
    stats_text = '\n'.join((
        r'$\bf{Statistics}$',
        r'Cosine Sim: $%.4f$' % (cos_sim, ),
        r'Conflict Ratio: $%.2f\%%$' % (conflict_ratio * 100, ),
        r'Params: $%d$' % (total_points, )
    ))
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='lightgray')
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # --- 7. 图例 ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Consistent',
               markerfacecolor='#2E8B57', markersize=8, alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='Conflict',
               markerfacecolor='#CD5C5C', markersize=8, alpha=0.7),
        Line2D([0], [0], color='green', linestyle='--', label='y=x'),
        Line2D([0], [0], color='red', linestyle=':', label='y=-x')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    
    # --- 8. 保存文件 (PDF) ---
    # 文件名格式: scatter_D1_D40_epoch0000_step000041.pdf
    save_name = f"scatter_{label_a}_vs_{label_b}_{step_info}.pdf"
    save_path = os.path.join(output_dir, save_name)
    
    plt.savefig(save_path, dpi=300)
    plt.close(fig) # 重要：关闭图像释放内存
    print(f"[Saved] {save_path} | Sim: {cos_sim:.3f}")


def batch_process_gradients(dir_a, dir_b, label_a="DatasetA", label_b="DatasetB", 
                            output_dir="./results_plots", filter_keyword=None):
    """
    批量处理两个目录下的梯度文件
    :param filter_keyword: 如果不为None，只处理文件名包含该字符串的文件 (例如 'epoch0000')
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 获取两个目录下的所有 .npy 文件名
    files_a = set([f for f in os.listdir(dir_a) if f.endswith('.npy')])
    files_b = set([f for f in os.listdir(dir_b) if f.endswith('.npy')])

    # 找到共有的文件 (交集)
    common_files = list(files_a.intersection(files_b))
    common_files.sort() # 排序，保证按step顺序处理

    # 过滤文件
    if filter_keyword:
        common_files = [f for f in common_files if filter_keyword in f]

    if not common_files:
        print("未找到共有的 .npy 文件，请检查路径或文件名是否一致。")
        return

    print(f"找到 {len(common_files)} 个匹配文件，准备开始绘图...")
    print(f"输出目录: {output_dir}")

    for f_name in common_files:
        path_a = os.path.join(dir_a, f_name)
        path_b = os.path.join(dir_b, f_name)
        
        plot_single_pair(
            path_a=path_a, 
            path_b=path_b, 
            label_a=label_a, 
            label_b=label_b, 
            output_dir=output_dir,
            file_name=f_name,
            max_points=200000  # 可以根据需要调整采样数
        )
    
    print("批量处理完成！")

# ================= 配置与执行 =================

# 1. 设置输入目录
dir_d1 = "/home/zhangchenyu/code/CSIGPT/results/D1/temporal_16batch"
dir_d40 = "/home/zhangchenyu/code/CSIGPT/results/D40/freq_16batch"

# 2. 执行批量绘图
# filter_keyword="epoch0000" 表示只处理 epoch 0 的文件
# 如果想处理所有 epoch，将 filter_keyword 设为 None
batch_process_gradients(
    dir_a=dir_d1, 
    dir_b=dir_d40, 
    label_a="D1_Temporal", 
    label_b="D40_Freq", 
    output_dir="./results/gradient_plots_pdf",
    filter_keyword="epoch0000" 
)