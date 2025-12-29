import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange
from transformers import BertConfig, BertModel
import matplotlib.pyplot as plt
import warnings

# 引入数据加载器 (假设在 util/data.py 中)
from models.baseline.CSIBERT import CSIBERT
from models.baseline.model import *
from util.data import data_load_baseline

warnings.filterwarnings("ignore")


def calculate_nmse(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse.item()

def preprocess_batch(H):
    """将 (B, T, K, U) 转换为 (B*T, K, U*2)"""
    B, T, K, U = H.shape
    H_real = H.real
    H_imag = H.imag
    H_combined = torch.stack([H_real, H_imag], dim=-1)
    H_combined = rearrange(H_combined, 'b t k u c -> (b t) k (u c)')
    return H_combined.float()

def apply_random_mask(inputs, mask_ratio=0.15, device='cuda'):
    batch_size, seq_len, feat_dim = inputs.shape
    # 生成随机掩码
    mask = torch.rand((batch_size, seq_len), device=device) < mask_ratio
    masked_inputs = inputs.clone()
    # 将被 Mask 的位置置为 0
    masked_inputs[mask.unsqueeze(-1).expand_as(inputs)] = 0
    # BERT 需要 attention mask (这里设为全1，表示没有 Padding)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    return masked_inputs, mask, attention_mask

# ==========================================
# 3. 训练与评估逻辑
# ==========================================

def train_baseline(model, train_loader, epochs=5, lr=1e-3, device='cuda', mask_ratio=0.15):
    """现场训练 MLP Baseline"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    print(f"Training Baseline: {model.__class__.__name__}...")
    for epoch in range(epochs):
        total_loss = 0
        for samples, _, _ in train_loader:
            samples = samples.to(device)
            clean_data = preprocess_batch(samples)
            # 对 Baseline 也应用同样的 Mask 机制进行训练
            masked_data, _, _ = apply_random_mask(clean_data, mask_ratio, device)
            
            optimizer.zero_grad()
            outputs = model(masked_data)
            loss = criterion(outputs, clean_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"  Epoch {epoch+1}/{epochs} Loss: {total_loss/len(train_loader):.6f}")
    return model

def evaluate_model(model, test_loader, device='cuda', mask_ratio=0.15, model_name="Model"):
    model.eval()
    nmse_list = []
    
    with torch.no_grad():
        for samples, _, _ in test_loader:
            samples = samples.to(device)
            clean_data = preprocess_batch(samples)
            
            # 测试时使用随机 Mask
            masked_data, _, attn_mask = apply_random_mask(clean_data, mask_ratio, device)
            
            if "BERT" in model_name:
                outputs = model(masked_data, attention_mask=attn_mask)
            else:
                outputs = model(masked_data) # MLP 不需要 attention mask
            
            batch_nmse = calculate_nmse(outputs, clean_data)
            nmse_list.append(batch_nmse)
            
    avg_nmse = np.mean(nmse_list)
    print(f"[{model_name}] Average NMSE: {avg_nmse:.6f}")
    return avg_nmse

# ==========================================
# 4. 主程序
# ==========================================

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_dir', default=None, type=str, help='Data directory')
    parser.add_argument('--dataset', default="D1", type=str)
    parser.add_argument('--checkpoint_path', default='./output_dir/bert4mimo_best.pth', help='Path to BERT checkpoint')
    parser.add_argument('--mask_ratio', default=0.15, type=float)
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    print("Loading Data...")
    dataset_train = data_load_baseline(args, dataset_type='train', data_num=1.0) 
    dataset_test = data_load_baseline(args, dataset_type='val')
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # 获取维度信息
    # 假设 data_load_baseline 返回的数据形状是 (B, T, K, U)
    # 我们需要取一个样本来确认 K (Seq Len) 和 U (Antenna)
    sample_batch, _, _ = next(iter(test_loader))
    B_raw, T_raw, K, U = sample_batch.shape
    
    feature_dim = U * 2 # 实部 + 虚部
    seq_len = K
    
    print(f"Data Info: Sequence Length (Subcarriers)={seq_len}, Feature Dim (Antennas*2)={feature_dim}")

    results = {}

    # ==========================================
    # 2. 评估 BERT4MIMO
    # ==========================================
    print("\n--- Evaluating BERT4MIMO ---")
    bert_model = CSIBERT(feature_dim=feature_dim).to(device)
    
    if os.path.isfile(args.checkpoint_path):
        print(f"Loading BERT checkpoint from {args.checkpoint_path}")
        bert_model.load_state_dict(torch.load(args.checkpoint_path))
    else:
        print(f"Warning: Checkpoint {args.checkpoint_path} not found! Using random weights for test.")

    bert_nmse = evaluate_model(bert_model, test_loader, device, args.mask_ratio, "BERT4MIMO")
    results['BERT4MIMO'] = bert_nmse

    # ==========================================
    # 3. 训练 & 评估 MLP Baseline
    # ==========================================
    print("\n--- Training & Evaluating MLP Baseline ---")
    
    # 论文中提到 Baseline 是 hidden layer of size 512 
    mlp_model = MLP(seq_len=seq_len, feature_dim=feature_dim, hidden_size=512).to(device)
    
    # 现场训练 5 个 epoch (由于 MLP 参数少，收敛快，5个 epoch 足够看到性能差异)
    mlp_model = train_baseline(mlp_model, train_loader, epochs=5, device=device, mask_ratio=args.mask_ratio)
    
    mlp_nmse = evaluate_model(mlp_model, test_loader, device, args.mask_ratio, "MLP")
    results['MLP'] = mlp_nmse

    # ==========================================
    # 4. 结果展示
    # ==========================================
    print("\n================ Results Summary ================")
    df = pd.DataFrame(list(results.items()), columns=['Model', 'NMSE'])
    df = df.sort_values(by='NMSE')
    print(df)
    
    # 保存结果图
    plt.figure(figsize=(6, 5))
    plt.bar(df['Model'], df['NMSE'], color=['#4285F4', '#EA4335'], width=0.5)
    plt.ylabel('NMSE (Lower is Better)')
    plt.title('Reconstruction NMSE: BERT4MIMO vs MLP')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    output_img = 'nmse_comparison.png'
    plt.savefig(output_img)
    print(f"Chart saved to {output_img}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)