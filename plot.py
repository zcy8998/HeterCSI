import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import seaborn as sns

# 设置绘图风格
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
    'legend.frameon': True,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def simulate_hetercsi_trends():
    """
    模拟 HeterCSI 泛化性能随桶数目、数据异构程度及数据集规模的变化趋势。
    """
    
    # 定义桶数目的范围 (X轴)
    buckets = np.arange(1, 61)
    
    # -------------------------------------------------------------------------
    # 模拟 1: 泛化性 vs. 桶数目 (U型曲线) & 异构程度的影响
    # -------------------------------------------------------------------------
    
    # 定义三个异构程度场景
    # Low Heterogeneity: 数据长度差异小，容易对齐
    # High Heterogeneity: 数据长度差异极大，难以对齐，需要更多桶
    
    def calculate_nmse_curve(b_vals, heterogeneity_factor):
        """
        构建合成的 NMSE 曲线模型
        NMSE = (梯度冲突带来的误差) + (随机性丢失带来的误差)
        """
        # 1. 梯度冲突项 (Gradient Conflict / Padding Cost)
        # 随桶数增加呈指数下降。异构性越高，下降越慢（需要更多桶才能降下来）。
        conflict_error = 0.6 * np.exp(-0.15 * b_vals / heterogeneity_factor)
        
        # 2. 随机性丢失项 (Diversity Loss / Overfitting Risk)
        # 随桶数增加而上升。桶越多，近似顺序训练的风险越大。
        # 这一项相对独立于数据异构性，主要取决于桶的数量与总数据量的比例
        diversity_loss = 0.00015 * (b_vals ** 2.2)
        
        # 基础底噪 (不可约误差)
        base_error = 0.1
        
        return conflict_error + diversity_loss + base_error

    # 生成曲线数据
    y_low_het = calculate_nmse_curve(buckets, heterogeneity_factor=0.5)
    y_med_het = calculate_nmse_curve(buckets, heterogeneity_factor=1.0)
    y_high_het = calculate_nmse_curve(buckets, heterogeneity_factor=2.0)

    # 找到最优桶数目 (最低点)
    opt_low = buckets[np.argmin(y_low_het)]
    opt_med = buckets[np.argmin(y_med_het)]
    opt_high = buckets[np.argmin(y_high_het)]

    # -------------------------------------------------------------------------
    # 模拟 2: 数据集规模的影响 (整体下移)
    # -------------------------------------------------------------------------
    # 假设在中等异构度下，增加数据集数量
    # 更多数据 -> 更好的泛化 -> 曲线整体向下平移
    y_data_8 = y_med_het + 0.15  # 8个数据集
    y_data_32 = y_med_het        # 32个数据集 (基准)

    # -------------------------------------------------------------------------
    # 绘图可视化
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 图 1: 异构程度对最优桶数的影响
    # 平滑曲线处理
    x_new = np.linspace(buckets.min(), buckets.max(), 300)
    
    spl_low = make_interp_spline(buckets, y_low_het)
    spl_med = make_interp_spline(buckets, y_med_het)
    spl_high = make_interp_spline(buckets, y_high_het)

    ax1.plot(x_new, spl_low(x_new), label='低异构性 (Low Heterogeneity)', color='green', linewidth=2.5)
    ax1.plot(x_new, spl_med(x_new), label='中异构性 (Medium Heterogeneity)', color='blue', linewidth=2.5)
    ax1.plot(x_new, spl_high(x_new), label='高异构性 (High Heterogeneity)', color='red', linewidth=2.5)

    # 标记最优点
    ax1.scatter(opt_low, np.min(y_low_het), color='green', s=100, zorder=5)
    ax1.text(opt_low, np.min(y_low_het)-0.02, f'Optimal B={opt_low}', ha='center', color='green', fontweight='bold')
    
    ax1.scatter(opt_med, np.min(y_med_het), color='blue', s=100, zorder=5)
    ax1.text(opt_med, np.min(y_med_het)-0.02, f'Optimal B={opt_med}', ha='center', color='blue', fontweight='bold')
    
    ax1.scatter(opt_high, np.min(y_high_het), color='red', s=100, zorder=5)
    ax1.text(opt_high, np.min(y_high_het)-0.02, f'Optimal B={opt_high}', ha='center', color='red', fontweight='bold')

    # 添加趋势箭头
    ax1.annotate('异构性越高，最优桶数越大', xy=(opt_low, np.min(y_low_het)+0.2), xytext=(opt_high, np.min(y_high_het)+0.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)

    ax1.set_title('数据异构程度对泛化性(NMSE)与最优桶数的影响', fontsize=14)
    ax1.set_xlabel('桶数目 (Number of Buckets)', fontsize=12)
    ax1.set_ylabel('泛化误差 (Zero-Shot NMSE)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图 2: 数据集规模的影响
    spl_d8 = make_interp_spline(buckets, y_data_8)
    spl_d32 = make_interp_spline(buckets, y_data_32)

    ax2.plot(x_new, spl_d8(x_new), label='数据集规模 = 8 (Small Scale)', color='orange', linestyle='--', linewidth=2.5)
    ax2.plot(x_new, spl_d32(x_new), label='数据集规模 = 32 (Large Scale)', color='purple', linewidth=2.5)

    # 添加箭头表示提升
    mid_x = 25
    y_start = spl_d8([mid_x])
    y_end = spl_d32([mid_x])
    ax2.annotate('数据规模增加带来泛化提升', xy=(mid_x, y_end), xytext=(mid_x, y_start),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', fontsize=12)

    ax2.set_title('数据集规模对泛化性能的影响', fontsize=14)
    ax2.set_xlabel('桶数目 (Number of Buckets)', fontsize=12)
    ax2.set_ylabel('泛化误差 (Zero-Shot NMSE)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("111")
    plt.show()

if __name__ == "__main__":
    simulate_hetercsi_trends()