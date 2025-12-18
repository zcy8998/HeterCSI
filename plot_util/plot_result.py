from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_nmse_results(result_file_path, target_task='random'):
    """
    读取NMSE结果文件并绘制指定任务的NMSE折线图
    
    参数:
    result_file_path (str): NMSE结果文件路径
    target_task (str): 要可视化的任务类型('random', 'temporal', 'freq')
    """
    # 1. 读取结果数据
    results = pd.read_csv(result_file_path)
    
    # 过滤目标任务的记录
    task_data = results[results['mask_type'] == target_task]
    
    if task_data.empty:
        print(f"错误: 文件中找不到任务'{target_task}'的数据")
        print(f"可用任务类型: {results['mask_type'].unique().tolist()}")
        return
    
    # 2. 创建图形
    plt.figure(figsize=(12, 6), dpi=150)
    
    # 3. 设置样式
    sns.set_theme(context='talk', style='whitegrid', font='serif')
    sns.set_palette('tab10')
    
    # 4. 创建主图
    ax = sns.lineplot(
        data=task_data, x='epoch', y='avg_nmse',
        marker='o', markersize=6, linewidth=1.5, 
        color='royalblue'
    )
    
    # 5. 添加标识线
    min_nmse = task_data['avg_nmse'].min()
    min_epoch = task_data.loc[task_data['avg_nmse'].idxmin(), 'epoch']
    
    plt.axhline(y=min_nmse, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=min_epoch, color='r', linestyle='--', alpha=0.7)
    
    # 6. 添加标签和注释
    plt.title(f'{target_task.capitalize()} Mask NMSE Performance', fontsize=18, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Normalized Mean Squared Error (NMSE)', fontsize=14)
    
    # 最佳点标注
    bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.9)
    plt.annotate(f'Best: {min_nmse:.4f}',
                 xy=(min_epoch, min_nmse),
                 xytext=(min_epoch + 0.05 * max(task_data['epoch']), min_nmse * 1.05),
                 bbox=bbox_props,
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle3", color='gray'))
    
    # 7. 自定义网格和边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 8. 添加背景区域
    plt.fill_between(
        x=task_data['epoch'],
        y1=task_data['avg_nmse'],
        y2=0,
        color='skyblue',
        alpha=0.2
    )
    
    # 9. 优化布局
    plt.tight_layout()
    
    # 10. 保存结果到源文件目录
    plot_file = Path(result_file_path).parent / f"{target_task}_mask_nmse.png"
    plt.savefig(plot_file, bbox_inches='tight', dpi=200)
    print(f"可视化结果已保存至: {plot_file}")
    
    # 显示图形
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 替换为你的结果文件路径
    result_file = "/path/to/your/results/nmse_results.csv"
    
    # 选择要可视化的任务类型 ('random', 'temporal', 'freq')
    plot_nmse_results(result_file, target_task='random')