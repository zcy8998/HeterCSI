import pdb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# 1. 读取验证结果数据
results_df = pd.read_csv('/mnt/4T/2/zcy/CSIGPT/4dataset_pretrain_csi_pretrain_random_motivation_2/validation_results.csv')
print(results_df)
# 2. 预处理数据
train_datasets = sorted(results_df['data_loader_idx'].unique())
test_datasets = sorted(results_df['dataset_name'].unique())
dataset_to_idx = {name: idx for idx, name in enumerate(test_datasets)}
train_dataset=["D1","D3","D6","D8"]
# test_dataset=["D1","D3","D6","D8"]
# 3. 构建性能矩阵
num_train = len(train_datasets)
num_test = len(test_datasets)
performance_matrix = np.zeros((num_train, num_test))

for train_dataset in train_datasets:
    train_data = results_df[results_df['data_loader_idx'] == train_dataset]
    last_epoch = train_data['epoch'].max()
    last_epoch_data = train_data[train_data['epoch'] == last_epoch]
    
    for _, row in last_epoch_data.iterrows():
        # pdb.set_trace()
        train_idx = dataset_to_idx[row['data_loader_idx']]
        test_idx = dataset_to_idx[row['dataset_name']]
        performance_matrix[train_idx, test_idx] = row['nmse']

# 4. 创建更精细的网格用于曲面插值
x = np.arange(1, num_train + 1)
y = np.arange(1, num_test + 1)
X, Y = np.meshgrid(x, y)
Z = performance_matrix.T

# 创建更密集的网格用于平滑曲面
xi = np.linspace(1, num_train, 100)
yi = np.linspace(1, num_test, 100)
XI, YI = np.meshgrid(xi, yi)

# 使用griddata进行插值
points = np.array([[x_val, y_val] for x_val, y_val in zip(X.flatten(), Y.flatten())])
values = Z.flatten()
ZI = griddata(points, values, (XI, YI), method='cubic')

# 5. 创建科研风格的3D可视化
# plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 绘制平滑曲面
surf = ax.plot_surface(XI, YI, ZI, cmap='viridis', 
                       rstride=2, cstride=2,
                       alpha=0.8, antialiased=True,
                       linewidth=0.1, edgecolor='k')

# 添加原始数据点
ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
           c='black', s=30, alpha=1.0, depthshade=False)

# 添加等高线投影
offset = Z.min() - 0.1 * (Z.max() - Z.min())
ax.contourf(XI, YI, ZI, zdir='z', offset=offset, 
            cmap='viridis', alpha=0.3, levels=20)

# 设置z轴范围
ax.set_zlim(offset, Z.max() + 0.05 * (Z.max() - Z.min()))

# 添加颜色条
cbar = fig.colorbar(surf, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('NMSE', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)

# 设置标签
ax.set_xlabel('Training Dataset', fontsize=12, labelpad=12)
ax.set_ylabel('Test Dataset', fontsize=12, labelpad=12)
ax.set_zlabel('NMSE', fontsize=12, labelpad=12)

# 设置刻度
ax.set_xticks(np.arange(1, num_train+1))
ax.set_yticks(np.arange(1, num_test+1))
ax.set_xticklabels([f"D{i}" for i in range(1, num_train+1)], fontsize=10)
ax.set_yticklabels([f"D{i}" for i in range(1, num_test+1)], fontsize=10)

# 优化视角
ax.view_init(elev=30, azim=-135)

# 添加网格和美化
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.xaxis.pane.set_alpha(0.05)
ax.yaxis.pane.set_alpha(0.05)
ax.zaxis.pane.set_alpha(0.05)
ax.grid(True, linestyle='--', alpha=0.4)

# 添加图例说明
# ax.text2D(0.05, 0.95, "Each point represents NMSE when\nmodel trained on DSx is tested on DSy", 
#           transform=ax.transAxes, fontsize=10, 
#           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig("motivation_4data_2.png", dpi=300, bbox_inches='tight')
plt.show()