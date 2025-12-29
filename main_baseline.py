import argparse
import datetime
import os
import h5py
import pdb
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from transformers import BertConfig, BertModel
from einops import rearrange

from models.baseline.CSIBERT import CSIBERT
from models.baseline.model import *
from util.data import *
# from util.misc import NativeScalerWithGradNormCount as NativeScaler # 暂时不用

# 忽略警告
import warnings
warnings.filterwarnings("ignore")
 

def get_args_parser():
    parser = argparse.ArgumentParser('BERT4MIMO Training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int) # 建议调大 Batch Size
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--model_type', default='bert4mimo', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--dataset', default="D1", type=str)
    parser.add_argument('--data_num', default=1.0, type=float)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--mask_ratio', default=0.15, type=float, help='Masking ratio for BERT')
    parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    return parser

class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        # NMSE calculation
        power = torch.sum(x ** 2)
        mse = torch.sum((x - x_hat) ** 2)
        nmse = mse / power
        return nmse

def save_best_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)

def preprocess_batch_for_bert(H):
    """
    将原始 (B, T, K, U) 的复数数据转换为 BERT 需要的 (B*T, K, U*2) 实数数据
    """
    # H: [B, T, K, U] (Complex)
    B, T, K, U = H.shape
    
    # 1. 拆分实虚部并归一化 (简单 Instance Norm 风格，防止数值过大)
    # 注意：这里在 Tensor 层面做简单的标准化，或者你可以依赖 dataset 的预处理
    H_real = H.real
    H_imag = H.imag
    
    # 简单归一化 (Optional, 根据你的数据分布决定是否保留)
    # mean = H_real.mean()
    # std = H_real.std() + 1e-6
    # H_real = (H_real - mean) / std
    # H_imag = (H_imag - mean) / std

    # 2. 堆叠: (B, T, K, U, 2)
    H_combined = torch.stack([H_real, H_imag], dim=-1)
    
    # 3. 展平 Batch 和 Time: (B*T, K, U, 2)
    # BERT4MIMO 处理的是单帧快照，所以 T 维度的每一帧都是一个独立的样本
    H_combined = rearrange(H_combined, 'b t k u c -> (b t) k (u c)')
    
    # 输出形状: (Batch_Size_Effective, Sequence_Length=K, Feature_Dim=U*2)
    return H_combined.float()

def apply_random_mask(inputs, mask_ratio=0.15, device='cuda'):
    """
    对输入的 CSI 矩阵进行随机掩码
    inputs: (B, K, Feat)
    """
    batch_size, seq_len, feat_dim = inputs.shape
    
    # 生成随机掩码矩阵 (B, K)
    # probability < mask_ratio 的地方设为 True (即被 Mask)
    mask = torch.rand((batch_size, seq_len), device=device) < mask_ratio
    
    # 创建被 Mask 后的输入
    masked_inputs = inputs.clone()
    
    # 将被 Mask 的位置（所有特征维度）置为 0
    # mask.unsqueeze(-1): (B, K, 1) -> 广播到 (B, K, Feat)
    masked_inputs[mask.unsqueeze(-1).expand_as(inputs)] = 0
    
    # 生成 Attention Mask (BERT 需要知道哪些是 Padding，这里没有 Padding，全是 1)
    # 但如果为了严谨，可以传入全 1 的 attention_mask
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    
    return masked_inputs, mask, attention_mask


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    os.makedirs(args.log_dir, exist_ok=True)

    # 加载数据集维度信息
    path = os.path.join(args.data_dir, args.dataset, "test_data.mat")
    with h5py.File(path, 'r') as f:
        dset = f[f'H_test']
        # 注意：你需要确认你的 .mat 文件存储顺序。
        # 代码原文写的是 U, K, T, 但处理时用的 samples (B, T, K, U)。
        # 这里假设读取出来的形状是 (U, K, T, Samples) 或者类似的，请根据你的实际数据调整。
        # 假设 dset.shape 对应 [U, K, T, _]
        U_dim, K_dim, T_dim, _ = dset.shape
        print(f"Data Shapes -> Antennas(U): {U_dim}, Subcarriers(K): {K_dim}, Time(T): {T_dim}")

    # 加载 Dataloader
    dataset_train = data_load_baseline(args, dataset_type='train', data_num=args.data_num)
    dataset_val = data_load_baseline(args, dataset_type='val')

    # 计算 Feature Dimension
    # 输入给 BERT 的特征维度 = 天线数 * 2 (实部+虚部)
    feature_dim = U_dim * 2 
    seq_len = K_dim
    
    # 初始化模型
    print(f"Initializing BERT4MIMO with Feature Dim: {feature_dim} (Sequence Len: {K_dim})")
    model = None
    if args.model_type == 'bert4mimo':
        model = CSIBERT(feature_dim=feature_dim, num_attention_heads=8).to(device)
    elif args.model_type == 'mlp':
        # MLP 初始化需要传入序列长度 (seq_len) 用于展平操作
        model = MLP(seq_len=seq_len, feature_dim=feature_dim, hidden_size=512).to(device)
    
    criterion = NMSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            model.load_state_dict(torch.load(args.resume))
        else:
            print("No checkpoint found.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters: {total_params / 1e6:.5f}M")
        
    print('Start training BERT4MIMO...')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        epoch_train_loss = []
        
        # ============ Training Loop =============== #
        model.train()
        for iteration, (samples, _, _) in enumerate(dataset_train, 1):
            # samples 形状: [Batch, Time, Subcarriers, Antennas] -> (B, T, K, U)
            # 或者是 (B, U, K, T) 取决于 data_load_baseline 的实现
            # 根据你之前的代码：B, T, K, U = samples.shape
            
            # 1. 数据转移到 GPU
            samples = samples.to(device) 
            
            # 2. 预处理：变为 (B*T, K, U*2)
            # 我们将所有时间步视为独立的 CSI 快照进行重建
            original_data = preprocess_batch_for_bert(samples) 
            
            # 3. 生成掩码数据
            masked_data, mask, attn_mask = apply_random_mask(original_data, mask_ratio=args.mask_ratio, device=device)
            
            optimizer.zero_grad()
            
            # 4. 模型前向传播
            # 输入: 带有 0 的 masked_data
            # 输出: 重建后的完整数据
            outputs = model(masked_data, attention_mask=attn_mask)
            
            # 5. 计算 Loss
            # BERT4MIMO 的论文通常计算全局重建 Loss (不仅是被 Mask 的部分)
            loss = criterion(outputs, original_data)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss.append(loss.item())

        t_loss = np.mean(epoch_train_loss)
        epoch_time = time.time() - start_time
        print('Epoch: {}/{} | Train NMSE: {:.7f} | Time: {:.2f}s'.format(epoch+1, args.epochs, t_loss, epoch_time)) 

        # ============ Validation Loop =============== #
        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for iteration, (samples, _, _) in enumerate(dataset_val, 1):
                samples = samples.to(device)
                
                # 预处理
                original_data = preprocess_batch_for_bert(samples)
                
                # 验证时同样进行 Mask，测试重建能力
                masked_data, mask, attn_mask = apply_random_mask(original_data, mask_ratio=args.mask_ratio, device=device)
                
                outputs = model(masked_data, attention_mask=attn_mask)
                loss = criterion(outputs, original_data)
                
                epoch_val_loss.append(loss.item())
                
            v_loss = np.mean(epoch_val_loss)
            print('Validate NMSE: {:.7f}'.format(v_loss))
            
            # 保存最佳模型
            if ((epoch + 1) % 10) == 0: # 每10个epoch保存一次，或根据 loss 保存
                save_path = os.path.join(args.output_dir, f"bert4mimo_epoch_{epoch+1}.pth")
                save_best_checkpoint(model, save_path)
                print(f"Saved checkpoint to {save_path}")

    total_time = time.time() - start_time
    print(f'Training finished. Total time: {str(datetime.timedelta(seconds=int(total_time)))}')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)