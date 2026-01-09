import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt
import warnings
import h5py

# 假设相关的模型定义和数据加载器在对应的路径中
# 请确保这些路径与你的项目结构一致
from models.baseline.CSIBERT import CSIBERT
from models.baseline.model import MLP 
from util.data import data_load_baseline

warnings.filterwarnings("ignore")

# ==========================================
# 1. 核心评估函数 (保持不变)
# ==========================================

def calculate_nmse_by_sample(x_hat, x, B, T):
    """
    按样本(N=Batch_Size)计算 NMSE。
    x_hat, x shape: (B*T, K, D)
    逻辑：将 T, K, D 全部作为特征维度求和，计算每个 Batch 条目的 NMSE 后取平均。
    """
    # 1. 恢复出 Batch 和 Time 维度 -> (B, T, K, D)
    x_hat = rearrange(x_hat, '(b t) k d -> b t k d', b=B, t=T)
    x = rearrange(x, '(b t) k d -> b t k d', b=B, t=T)
    
    # 2. 定义求和维度：对每个样本内部的所有 Time, Seq_Len, Feature 维度求和
    # 执行后 shape 将变为 (B,)
    reduce_dims = (1, 2, 3)
    
    # 3. 计算每个 Sample 的误差平方和与功率
    mse_per_sample = torch.sum((x - x_hat) ** 2, dim=reduce_dims)
    power_per_sample = torch.sum(x ** 2, dim=reduce_dims)
    
    # 4. 计算每个 Sample 的 NMSE (加上 eps 防止除以 0)
    nmse_per_sample = mse_per_sample / (power_per_sample + 1e-12)
    
    # 5. 返回该 Batch 内 N 个样本的平均 NMSE
    return torch.mean(nmse_per_sample).item()

def preprocess_batch(H):
    """将 (B, T, K, U) 转换为 (B*T, K, U*2)"""
    B, T, K, U = H.shape
    H_real = H.real
    H_imag = H.imag
    H_combined = torch.stack([H_real, H_imag], dim=-1)
    # 合并 Batch 和 Time 维度供模型处理
    H_combined = rearrange(H_combined, 'b t k u c -> (b t) k (u c)')
    return H_combined.float()

def apply_random_mask(inputs, mask_ratio=0.15, device='cuda'):
    batch_size, seq_len, feat_dim = inputs.shape
    mask = torch.rand((batch_size, seq_len), device=device) < mask_ratio
    masked_inputs = inputs.clone()
    masked_inputs[mask.unsqueeze(-1).expand_as(inputs)] = 0
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    return masked_inputs, mask, attention_mask

def evaluate_model(model, test_loader, device='cuda', mask_ratio=0.15, model_name="Model"):
    model.eval()
    nmse_accum = 0.0
    num_batches = 0
    
    print(f"Testing {model_name}...")
    
    with torch.no_grad():
        for samples, _, _ in test_loader:
            # 获取原始维度 B 和 T
            B, T, K, U = samples.shape
            samples = samples.to(device)
            
            # 预处理数据
            clean_data = preprocess_batch(samples)
            masked_data, _, attn_mask = apply_random_mask(clean_data, mask_ratio, device)
            
            # 推理
            if "BERT" in model_name.upper():
                outputs = model(masked_data, attention_mask=attn_mask)
            else:
                outputs = model(masked_data)
            
            # 计算 Batch 平均 NMSE
            batch_avg_nmse = calculate_nmse_by_sample(outputs, clean_data, B, T)
            
            # 累加 (因为 batch_size 可能不一致，严谨的话应该乘 B 再除总数，这里简化为对 batch 平均值求平均)
            nmse_accum += batch_avg_nmse
            num_batches += 1
            
    avg_nmse = nmse_accum / num_batches
    print(f"[{model_name}] Average NMSE (per sample): {avg_nmse:.6f}")
    return avg_nmse

# ==========================================
# 2. 参数解析与主函数
# ==========================================

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_dir', default=None, type=str, help='Data directory')
    parser.add_argument('--dataset', default="D1", type=str, help='comma separated datasets, e.g. D1,D2')
    parser.add_argument('--mask_ratio', default=0.15, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume checkpoint path prefix')
    parser.add_argument('--model_type', default='bert4mimo', type=str)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader.')
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 1. 初始化结果文件
    result_file_path = os.path.join(args.log_dir, f"{args.model_type}_evaluation_results_re.txt")
    with open(result_file_path, 'w') as f:
        f.write(f"Model: {args.model_type}, CSI Reconstruction\n")
        f.write("Dataset, NMSE (dB)\n")

    # 2. 解析数据集列表
    dataset_list = args.dataset.split(',')
    all_nmse_linear = [] # 用于存储所有数据集的线性结果，以便计算总平均

    # ============ 循环测试每个数据集 ============ #
    for dataset_name in dataset_list:
        dataset_name = dataset_name.strip()
        print(f"\n{'='*30}\nProcessing Dataset: {dataset_name}\n{'='*30}")
        
        # 更新 dataset 参数
        args.dataset = dataset_name

        # 3. 加载数据
        print("Loading Dataset...")
        try:
            dataset_test = data_load_baseline(args, dataset_type='val')
            if dataset_test is None:
                print(f"Skipping {dataset_name} due to load error.")
                continue
            test_loader = dataset_test
            
            # 探测维度 (用于初始化模型)
            sample_batch, _, _ = next(iter(test_loader))
            B_raw, T_raw, K, U = sample_batch.shape
            feature_dim = U * 2
            seq_len = K
            print(f"Data Info -> Seq Length: {seq_len}, Feature Dim: {feature_dim}")
            
        except Exception as e:
            print(f"Error preparing data for {dataset_name}: {e}")
            continue

        # 4. 初始化模型 (根据当前数据集维度)
        if args.model_type == 'bert4mimo':
            model = CSIBERT(feature_dim=feature_dim).to(device)
        elif args.model_type == 'mlp':
            model = MLP(seq_len=seq_len, feature_dim=feature_dim, hidden_size=512).to(device)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        if args.resume:
            # 尝试直接使用 args.resume (如果它是完整文件路径)
            if os.path.isfile(args.resume):
                checkpoint_path = args.resume
            else:
                # 否则构造路径
                # 假设后缀是 _fullshot_bert 或类似的，这里暂时用 _fullshot
                # 你需要根据你的 save_dir 命名规则修改这个后缀
                checkpoint_path = f"{args.resume}{args.model_type}_{dataset_name}_fullshot_re/bert4mimo_epoch_150.pth"
                
                # 如果找不到，尝试一下通常的命名规则，例如 checkpoint-xxx.pth
                if not os.path.exists(checkpoint_path):
                     # 备选：尝试找目录下最新的 pth
                     pass

            if os.path.isfile(checkpoint_path):
                print(f"=> Loading weights from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                # 处理可能的 module. 前缀 (DataParallel)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace("module.", "")
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict, strict=False)
            else:
                print(f"=> [Warning] Checkpoint not found at: {checkpoint_path}")
                print("=> Testing with random weights (Results will be meaningless).")

        # 6. 执行评估
        nmse_linear = evaluate_model(model, test_loader, device, args.mask_ratio, args.model_type)
        nmse_db = 10 * np.log10(np.clip(nmse_linear, 1e-10, None))

        print(f"Dataset: {dataset_name} | NMSE: {nmse_linear:.6f} | dB: {nmse_db:.4f}")

        # 7. 写入结果到文件
        with open(result_file_path, 'a') as f:
            f.write(f"{dataset_name}, {nmse_db:.4f} dB\n")
        
        all_nmse_linear.append(nmse_linear)

    # ============ 所有数据集结束，计算平均值 ============ #
    if len(all_nmse_linear) > 0:
        # 计算平均 Linear NMSE，再转 dB
        avg_linear = np.mean(all_nmse_linear)
        avg_db = 10 * np.log10(np.clip(avg_linear, 1e-10, None))
        
        print("\n" + "="*30)
        print(f"All Tasks Finished.")
        print(f"Datasets: {dataset_list}")
        print(f"Average Linear NMSE: {avg_linear:.6f}")
        print(f"Average NMSE (dB): {avg_db:.4f} dB")
        print("="*30)
        
        # 写入平均值
        with open(result_file_path, 'a') as f:
            f.write(f"Avg, {avg_db:.4f} dB\n")
            
    print(f"Results saved to: {result_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)