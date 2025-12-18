import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# 设置科研风格的绘图参数
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(14, 8))

# 模拟数据 - 样本实际长度（48个样本）
np.random.seed(42)  # 确保结果可重现
# values = [192] * 12 + [376] * 12 + [768] * 12 + [960] * 12
values1 = [192] * 12 + [376] * 12
values2 = [768] * 12 + [960] * 12
sample_lengths1 = np.random.permutation(values1)
sample_lengths2 = np.random.permutation(values2)
sample_lengths = np.concatenate((sample_lengths1, sample_lengths2))

# 随机打乱数组
np.random.seed(42)  # 设置随机种子以确保结果可重现
# sample_lengths = np.random.permutation(values)

# 批次信息
batch_size = 8
num_batches = 6

# 计算每个批次的最大长度
batch_max_lengths = []
for i in range(num_batches):
    batch_samples = sample_lengths[i * batch_size:(i + 1) * batch_size]
    batch_max = np.max(batch_samples)
    batch_max_lengths.append(batch_max)

# 计算零填充部分和统计信息
padding_percent = 37.29
avg_batch_len = 737.00

# 创建堆叠柱状图
x = np.arange(48)
bars = ax.bar(x, sample_lengths, color='#2E86AB', edgecolor='white', linewidth=0.5, label='Sample Length')

# 添加红色虚线框表示每个批次的最大值
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size - 1
    batch_max = batch_max_lengths[i]

    # 创建红色虚线矩形框
    rect = patches.Rectangle(
        (start_idx - 0.5, 0), batch_size, batch_max,
        linewidth=2, edgecolor='#A23B72', linestyle='--', facecolor='none', alpha=0.7
    )
    ax.add_patch(rect)

# 设置坐标轴标签和标题
ax.set_xlabel('Sample Index (48 samples)', fontsize=14, labelpad=10)
ax.set_ylabel('Sample Length', fontsize=14, labelpad=10)

# 设置x轴刻度
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xticklabels(['0', '10', '20', '30', '40'])

# 移除y轴刻度
ax.set_yticks([])

# 添加网格线（只在x轴方向）
ax.grid(True, axis='x', alpha=0.3)

# 添加图例
# ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig("results_fig/zero_padding.png", dpi=300)