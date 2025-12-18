import numpy as np

import torch
import torch.fft as fft


def reconstruct_complex(x):
    # 将输入的实部/虚部组合的表示转换为复数形式
    # x的形状: [B, L, 2*P^3] 其中P是patch_size
    B, L, D = x.shape
    # 分割实部和虚部
    real_part = x[..., :D//2]
    imag_part = x[..., D//2:]
    # 组合成复数
    return torch.complex(real_part, imag_part)

def to_delay_doppler(x, input_size, patch_size):
    # x的形状: [B, L, P^3] 复数
    B, L, D = x.shape
    
    # 首先将数据还原为原始CSI形状 [B, T, K, U]
    # 这里需要知道原始维度信息，假设存储在self.original_dims中
    T, K, U = input_size
    
    # 计算每个维度的块数
    t_blocks = T // patch_size
    k_blocks = K // patch_size
    u_blocks = U // patch_size
    
    # 重塑为 [B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size]
    x = x.view(B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size)
    
    # 调整维度顺序为原始CSI顺序 [B, T, K, U]
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.view(B, T, K, U)
    
    # 应用2D傅里叶变换到子载波和天线维度（转换为时延-多普勒域）
    # 对K和U维度进行FFT
    x_dd = fft.ifft2(x, dim=(-2, -1))
    return x_dd