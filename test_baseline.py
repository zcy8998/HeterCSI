# import argparse
# import os
# import time
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from einops import rearrange
# from transformers import BertConfig, BertModel
# import matplotlib.pyplot as plt
# import warnings

# # 引入数据加载器 (假设在 util/data.py 中)
# from models.baseline.CSIBERT import CSIBERT
# from models.baseline.model import *
# from util.data import data_load_baseline

# warnings.filterwarnings("ignore")


# def calculate_nmse(x_hat, x):
#     power = torch.sum(x ** 2)
#     mse = torch.sum((x - x_hat) ** 2)
#     nmse = mse / power
#     return nmse.item()

# def preprocess_batch(H):
#     """将 (B, T, K, U) 转换为 (B*T, K, U*2)"""
#     B, T, K, U = H.shape
#     H_real = H.real
#     H_imag = H.imag
#     H_combined = torch.stack([H_real, H_imag], dim=-1)
#     H_combined = rearrange(H_combined, 'b t k u c -> (b t) k (u c)')
#     return H_combined.float()

# def apply_random_mask(inputs, mask_ratio=0.15, device='cuda'):
#     batch_size, seq_len, feat_dim = inputs.shape
#     # 生成随机掩码
#     mask = torch.rand((batch_size, seq_len), device=device) < mask_ratio
#     masked_inputs = inputs.clone()
#     # 将被 Mask 的位置置为 0
#     masked_inputs[mask.unsqueeze(-1).expand_as(inputs)] = 0
#     # BERT 需要 attention mask (这里设为全1，表示没有 Padding)
#     attention_mask = torch.ones((batch_size, seq_len), device=device)
#     return masked_inputs, mask, attention_mask

# # ==========================================
# # 3. 训练与评估逻辑
# # ==========================================

# def train_baseline(model, train_loader, epochs=5, lr=1e-3, device='cuda', mask_ratio=0.15):
#     """现场训练 MLP Baseline"""
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#     model.train()
    
#     print(f"Training Baseline: {model.__class__.__name__}...")
#     for epoch in range(epochs):
#         total_loss = 0
#         for samples, _, _ in train_loader:
#             samples = samples.to(device)
#             clean_data = preprocess_batch(samples)
#             # 对 Baseline 也应用同样的 Mask 机制进行训练
#             masked_data, _, _ = apply_random_mask(clean_data, mask_ratio, device)
            
#             optimizer.zero_grad()
#             outputs = model(masked_data)
#             loss = criterion(outputs, clean_data)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         # print(f"  Epoch {epoch+1}/{epochs} Loss: {total_loss/len(train_loader):.6f}")
#     return model

# def evaluate_model(model, test_loader, device='cuda', mask_ratio=0.15, model_name="Model"):
#     model.eval()
#     nmse_list = []
    
#     with torch.no_grad():
#         for samples, _, _ in test_loader:
#             samples = samples.to(device)
#             clean_data = preprocess_batch(samples)
            
#             # 测试时使用随机 Mask
#             masked_data, _, attn_mask = apply_random_mask(clean_data, mask_ratio, device)
            
#             if "BERT" in model_name:
#                 outputs = model(masked_data, attention_mask=attn_mask)
#             else:
#                 outputs = model(masked_data) # MLP 不需要 attention mask
            
#             batch_nmse = calculate_nmse(outputs, clean_data)
#             nmse_list.append(batch_nmse)
            
#     avg_nmse = np.mean(nmse_list)
#     print(f"[{model_name}] Average NMSE: {avg_nmse:.6f}")
#     return avg_nmse

# # ==========================================
# # 4. 主程序
# # ==========================================

# def get_args_parser():
#     parser = argparse.ArgumentParser('Evaluation', add_help=False)
#     parser.add_argument('--batch_size', default=32, type=int)
#     parser.add_argument('--data_dir', default=None, type=str, help='Data directory')
#     parser.add_argument('--dataset', default="D1", type=str)
#     parser.add_argument('--mask_ratio', default=0.15, type=float)
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--resume', default='', help='resume from checkpoint')
#     parser.add_argument('--model_type', default='bert4mimo', type=str)
#     parser.add_argument('--output_dir', default='./output_dir')
#     parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
#     parser.add_argument('--num_workers', default=4, type=int)
#     parser.add_argument('--pin_mem', action='store_true',
#                     help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
#     return parser

# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 1. 加载数据
#     print("Loading Data...")
#     dataset_train = data_load_baseline(args, dataset_type='train', data_num=1.0) 
#     dataset_test = data_load_baseline(args, dataset_type='val')
    
#     train_loader = dataset_train
#     test_loader = dataset_test

#     # 获取维度信息
#     # 假设 data_load_baseline 返回的数据形状是 (B, T, K, U)
#     # 我们需要取一个样本来确认 K (Seq Len) 和 U (Antenna)
#     sample_batch, _, _ = next(iter(test_loader))
#     B_raw, T_raw, K, U = sample_batch.shape
    
#     feature_dim = U * 2 # 实部 + 虚部
#     seq_len = K
    
#     print(f"Data Info: Sequence Length (Subcarriers)={seq_len}, Feature Dim (Antennas*2)={feature_dim}")

#     results = {}

#     # ==========================================
#     # 2. 评估 BERT4MIMO
#     # ==========================================
#     print("\n--- Evaluating BERT4MIMO ---")
#     bert_model = CSIBERT(feature_dim=feature_dim).to(device)
    
#     if os.path.isfile(args.resume):
#         print(f"Loading BERT checkpoint from {args.resume}")
#         bert_model.load_state_dict(torch.load(args.resume))
#     else:
#         print(f"Warning: Checkpoint {args.resume} not found! Using random weights for test.")

#     bert_nmse = evaluate_model(bert_model, test_loader, device, args.mask_ratio, "BERT4MIMO")
#     results['BERT4MIMO'] = bert_nmse

#     # ==========================================
#     # 3. 训练 & 评估 MLP Baseline
#     # ==========================================
#     print("\n--- Training & Evaluating MLP Baseline ---")
    
#     # 论文中提到 Baseline 是 hidden layer of size 512 
#     # mlp_model = MLP(seq_len=seq_len, feature_dim=feature_dim, hidden_size=512).to(device)
    
#     # # 现场训练 5 个 epoch (由于 MLP 参数少，收敛快，5个 epoch 足够看到性能差异)
#     # mlp_model = train_baseline(mlp_model, train_loader, epochs=5, device=device, mask_ratio=args.mask_ratio)
    
#     # mlp_nmse = evaluate_model(mlp_model, test_loader, device, args.mask_ratio, "MLP")
#     # results['MLP'] = mlp_nmse

#     # ==========================================
#     # 4. 结果展示
#     # ==========================================
#     print("\n================ Results Summary ================")
#     df = pd.DataFrame(list(results.items()), columns=['Model', 'NMSE'])
#     df = df.sort_values(by='NMSE')
#     print(df)
    
#     # 保存结果图
#     plt.figure(figsize=(6, 5))
#     plt.bar(df['Model'], df['NMSE'], color=['#4285F4', '#EA4335'], width=0.5)
#     plt.ylabel('NMSE (Lower is Better)')
#     plt.title('Reconstruction NMSE: BERT4MIMO vs MLP')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     output_img = 'nmse_comparison.png'
#     plt.savefig(output_img)
#     print(f"Chart saved to {output_img}")

# if __name__ == '__main__':
#     args = get_args_parser()
#     args = args.parse_args()
#     main(args)


import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt
import warnings

# 假设相关的模型定义和数据加载器在对应的路径中
from models.baseline.CSIBERT import CSIBERT
from models.baseline.model import MLP 
from util.data import data_load_baseline

warnings.filterwarnings("ignore")

# ==========================================
# 1. 核心评估函数
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
    nmse_list = []
    
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
            
            # 使用修正后的 NMSE 计算逻辑 (N = Batch_Size)
            batch_avg_nmse = calculate_nmse_by_sample(outputs, clean_data, B, T)
            nmse_list.append(batch_avg_nmse)
            
    avg_nmse = np.mean(nmse_list)
    print(f"[{model_name}] Average NMSE (per sample): {avg_nmse:.6f}")
    return avg_nmse

# ==========================================
# 2. 参数解析与主函数
# ==========================================

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_dir', default=None, type=str, help='Data directory')
    parser.add_argument('--dataset', default="D1", type=str)
    parser.add_argument('--mask_ratio', default=0.15, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--model_type', default='bert4mimo', type=str)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载数据
    print("Loading Dataset...")
    dataset_test = data_load_baseline(args, dataset_type='val')
    test_loader = dataset_test

    # 探测维度
    sample_batch, _, _ = next(iter(test_loader))
    B_raw, T_raw, K, U = sample_batch.shape
    feature_dim = U * 2
    seq_len = K
    print(f"Sequence Length: {seq_len}, Feature Dim: {feature_dim}")

    # 2. 根据 model_type 创建模型
    if args.model_type == 'bert4mimo':
        model = CSIBERT(feature_dim=feature_dim).to(device)
    elif args.model_type == 'mlp':
        model = MLP(seq_len=seq_len, feature_dim=feature_dim, hidden_size=512).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # 3. 加载预训练权重
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading weights from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # 兼容不同的保存格式
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        print("Warning: No valid checkpoint found, evaluating with random weights.")

    # 4. 执行评估
    nmse_result = evaluate_model(model, test_loader, device, args.mask_ratio, args.model_type)
    nmse_db = 10 * np.log10(np.clip(nmse_result, 1e-10, None))

    # 5. 输出简报
    print("\n" + "="*30)
    print(f"Evaluation Finished")
    print(f"Model: {args.model_type}")
    print(f"Final NMSE: {nmse_result:.8f}")
    print(f"NMSE (dB): {nmse_db:.4f}")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)