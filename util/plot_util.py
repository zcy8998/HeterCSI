import pdb
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class CSIVisualizer:
    """
    CSIVisualizer：用于加载 CSI 数据并绘制幅度曲线与热图。
    Attributes:
        csi (np.ndarray): 形状为 (num_times, num_subcarriers) 的复数 CSI 矩阵。
        amplitude (np.ndarray): CSI 幅度矩阵，形状同 csi。
    """

    def __init__(self, csi_data: np.ndarray):
        """
        初始化 CSIVisualizer。
        参数:
            csi_data (np.ndarray): 复数 CSI 数据矩阵。
        """
        if not isinstance(csi_data, np.ndarray):
            raise TypeError("csi_data 必须是 NumPy 数组")
        if csi_data.ndim != 2 or not np.iscomplexobj(csi_data):
            raise ValueError("csi_data 必须是二维复数数组，形状为 (时间点, 子载波数)")
        self.csi = csi_data
        self.amplitude = np.abs(self.csi)

    def plot_amplitude_curve(self, time_idx: int, figsize=(8, 4), marker='o'):
        """
        绘制指定时刻子载波的 CSI 幅度曲线。
        参数:
            time_idx (int): OFDM 符号索引（时间点）。
            figsize (tuple): 图像尺寸，默认为 (8, 4)。
            marker (str): 曲线标记符号，默认为 'o'。
        """
        num_times, num_subcarriers = self.amplitude.shape
        if not (0 <= time_idx < num_times):
            raise IndexError(f"time_idx 必须在 [0, {num_times-1}] 范围内")
        
        plt.figure(figsize=figsize)
        plt.plot(
            np.arange(num_subcarriers),
            self.amplitude[time_idx, :],
            marker=marker
        )
        plt.title(f'CSI Amplitude at OFDM Symbol {time_idx}')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_amplitude_heatmap(self, figsize=(10, 6), cmap='viridis'):
        """
        绘制 CSI 幅度热图（子载波 vs 时间）。
        参数:
            figsize (tuple): 图像尺寸，默认为 (10, 6)。
            cmap (str): 颜色映射，默认为 'viridis'。
        """
        num_times, num_subcarriers = self.amplitude.shape
        
        plt.figure(figsize=figsize)
        plt.imshow(
            self.amplitude.T,
            aspect='auto',
            origin='lower',
            extent=[0, num_times-1, 0, num_subcarriers-1],
            cmap=cmap
        )
        plt.colorbar(label='Amplitude')
        plt.title('CSI Amplitude Heatmap')
        plt.xlabel('Time (OFDM Symbol Index)')
        plt.ylabel('Subcarrier Index')
        plt.tight_layout()
        plt.show()


class CSIAttnVisualizer:
    def __init__(self, model, device='cuda'):
        """
        初始化注意力可视化器
        
        参数:
        model: 要可视化的模型
        device: 计算设备 ('cuda' 或 'cpu')
        input_size: 输入尺寸元组 (T, K, U)
        """
        self.model = model.to(device).eval()
        self.device = device

    def get_attention_distribution(self, input_data, block_index=-1):

        attn_file = None
        with torch.no_grad():
            # 应用所有预处理层（patch embedding + 位置编码）
            tokens = self.model.patch_embed(input_data)
            tokens = tokens + self.model.pos_embed[:, :tokens.size(1), :]
            
            save_attn = False
            # 通过所有block直到目标block
            for i in range(len(self.model.blocks) - 1):
                if i == block_index:
                    save_attn = True
                    attn_file = self.model.blocks[i](tokens, save_attn=save_attn)
                else:
                    tokens = self.model.blocks[i](tokens, save_attn=save_attn)
            
            if block_index == -1:
                # 如果block_index超出范围，使用最后一个block
                pdb.set_trace()
                attn_file = self.model.blocks[i](tokens, save_attn=True)
        
        return attn_file

    def visualize_attention(
        self,
        samples,
        save_path = None,
        cmap = "viridis",
        query_idx = None,
    ):
        """
        可视化注意力权重
        Args:
            attn_file: 注意力权重文件路径 (.npy)
            save_path: 图片保存路径 (None则显示图片)
            cmap: 颜色映射
            query_idx: 指定查询位置索引 (None显示全局)
            head_idx: 指定注意力头索引 (None则平均所有头)
            show_cls: 是否显示CLS token的注意力
        """

        attn_file = self.get_attention_distribution(samples)
        print(attn_file)

        pdb.set_trace()
        attn = np.load(attn_file)
        H, W = int(np.sqrt(attn.shape[0])), int(np.sqrt(attn.shape[0]))
        
        
        # # 处理指定查询位置
        # if query_idx is not None:
        #     attn = attn[query_idx]
        #     attn = attn.reshape(H, W)
        # else:
        #     # 平均所有查询位置
        #     attn = attn.mean(0).reshape(H, W)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(attn, cmap=cmap)
        plt.colorbar()
        plt.title(f"Attention Map\n{os.path.basename(attn_file)}")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

    # # visualize_global_attention 方法保持不变
    # def visualize_global_attention(self, input_data, spatial_coord, save_path='csi_attn_global.png', 
    #                               show=False, block_index=-1):
    #     # 获取注意力分布
    #     attn_grid = self.get_attention_distribution(input_data, spatial_coord, block_index)
        
    #     # 提取参考点所在维度的全局注意力切片
    #     t0, k0, u0 = spatial_coord
        
    #     # 创建2x2子图布局
    #     fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
    #     # T-K平面切片 (固定U=u0)
    #     im1 = axs[0, 0].imshow(attn_grid[:, :, u0], cmap='viridis', origin='lower')
    #     axs[0, 0].scatter([k0], [t0], c='red', marker='x', s=100)
    #     axs[0, 0].set_title(f"T-K Plane (U={u0})")
    #     axs[0, 0].set_xlabel('K')
    #     axs[0, 0].set_ylabel('T')
    #     fig.colorbar(im1, ax=axs[0, 0])
        
    #     # T-U平面切片 (固定K=k0)
    #     im2 = axs[0, 1].imshow(attn_grid[:, k0, :], cmap='viridis', origin='lower')
    #     axs[0, 1].scatter([u0], [t0], c='red', marker='x', s=100)
    #     axs[0, 1].set_title(f"T-U Plane (K={k0})")
    #     axs[0, 1].set_xlabel('U')
    #     axs[0, 1].set_ylabel('T')
    #     fig.colorbar(im2, ax=axs[0, 1])
        
    #     # K-U平面切片 (固定T=t0)
    #     im3 = axs[1, 0].imshow(attn_grid[t0, :, :], cmap='viridis', origin='lower')
    #     axs[1, 0].scatter([u0], [k0], c='red', marker='x', s=100)
    #     axs[1, 0].set_title(f"K-U Plane (T={t0})")
    #     axs[1, 0].set_xlabel('U')
    #     axs[1, 0].set_ylabel('K')
    #     fig.colorbar(im3, ax=axs[1, 0])
        
    #     # 3D散点图
    #     ax = axs[1, 1]
    #     ax = fig.add_subplot(2, 2, 4, projection='3d')
        
    #     # 采样点以减少计算量
    #     T, K, U = self.input_size
    #     step = max(1, T//20, K//20, U//20)
    #     t_idx, k_idx, u_idx = np.meshgrid(
    #         np.arange(0, T, step),
    #         np.arange(0, K, step),
    #         np.arange(0, U, step),
    #         indexing='ij'
    #     )
        
    #     # 获取采样点的注意力值
    #     values = attn_grid[t_idx, k_idx, u_idx].flatten()
        
    #     # 绘制3D散点图
    #     sc = ax.scatter(
    #         u_idx.flatten(), k_idx.flatten(), t_idx.flatten(), 
    #         c=values, cmap='viridis', alpha=0.6, s=10
    #     )
        
    #     # 标记参考点
    #     ax.scatter([u0], [k0], [t0], c='red', s=200, marker='*')
        
    #     ax.set_xlabel('U')
    #     ax.set_ylabel('K')
    #     ax.set_zlabel('T')
    #     ax.set_title('3D Attention Distribution')
    #     fig.colorbar(sc, ax=ax, label='Attention Weight')
        
    #     # 整体标题
    #     block_name = f"Block {block_index}" if block_index >= 0 else "Last Block"
    #     fig.suptitle(f"Global Attention from Point (T={t0}, K={k0}, U={u0}) - {block_name}", fontsize=16)
        
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    #     if show:
    #         plt.show()
    #     plt.close()
    #     print("Saved:", save_path)
        
    #     # 返回注意力网格和参考点信息
    #     return attn_grid, spatial_coord
