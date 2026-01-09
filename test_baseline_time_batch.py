# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# SpectralGPT: https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT
# --------------------------------------------------------
import argparse
import datetime
import os
import pdb
import time
from pathlib import Path
import h5py
import numpy as np

pdb.set_trace = lambda *args, **kwargs: None

import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch import nn
from einops import rearrange

# 假设这些模型和工具函数在你的项目中可以正常 import
from models.baseline.model import LSTM, Informer # 根据你的文件结构调整
from models.baseline.LLM4CP import Model as LLM4CP
from models.baseline.PAD import PAD3
from util.data import * 
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import hdf5storage

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU')

    # Model parameters
    parser.add_argument('--model_type', default='lstm', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task_type', default='temporal', type=str, help='Name of task to test')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint path prefix')
    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--dataset', default="D1", type=str,
                        help='dataset used to train (can be comma separated e.g. D1,D2)')
    parser.add_argument('--data_dir', default=None, type=str,
                        help='the data dir used to train')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')

    return parser


def NMSE_cuda(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


def LoadBatch_ofdm_1(H):
    # H: B,T,mul     [tensor complex]
    # out:B,T,mul*2  [tensor real]
    B, T, mul = H.shape
    H_real = np.zeros([B, T, mul, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm_2(H):
    # H: B,T,K,mul     [tensor complex]
    # out:B,T,K,mul*2  [tensor real]
    B, T, K, mul = H.shape
    H_real = np.zeros([B, T, K, mul, 2])
    H_real[:, :, :, :, 0] = H.real
    H_real[:, :, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, K, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cpu")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    os.makedirs(args.log_dir, exist_ok=True)
    args.distributed = False

    # [New]: 1. 准备结果文件
    result_file_path = os.path.join(args.log_dir, f"{args.model_type}_evaluation_results_time.txt")
    with open(result_file_path, 'w') as f:
        f.write(f"Task Type: {args.model_type} (Temporal Prediction)\n")
        f.write("Dataset, NMSE (dB)\n")

    # [New]: 2. 解析数据集列表
    dataset_list = args.dataset.split(',')
    
    # 用于存储所有数据集的【线性】NMSE，最后计算平均 dB
    all_nmse_linear = []

    # ============ 循环测试每个数据集 ============ #
    for dataset_name in dataset_list:
        dataset_name = dataset_name.strip()
        print(f"\n{'='*20} Testing Dataset: {dataset_name} {'='*20}")

        args.dataset = dataset_name # 更新 args 供 data_load_baseline 使用
        
        # [Step 1]: 读取维度 (U, K, T)
        path = os.path.join(args.data_dir, dataset_name, "test_data.mat")
        try:
            with h5py.File(path, 'r') as f:
                dset = f[f'H_test']
                U, K, T, _ = dset.shape
                print(f"[{dataset_name}] Dimensions -> U: {U}, K: {K}, T: {T}")
                t, k, u = T, K, U
        except Exception as e:
            print(f"Error loading dimensions for {dataset_name}: {e}")
            continue

        # [Logic]: 保持时域预测逻辑 (切分 T，输入特征维度跟 K 有关)
        pred_len = int(t / 2)
        
        # 加载数据
        dataset_test = data_load_baseline(args, dataset_type='test')
        if dataset_test is None: continue

        if args.model_type == 'lstm':
            model = LSTM(features=2*k, input_size=2*k, hidden_size=4*k, num_layers=4).to(device)
        elif args.model_type == 'llm4cp':
            model = LLM4CP(pred_len=pred_len, prev_len=pred_len, K=k, UQh=1, UQv=1, BQh=1, BQv=1).to(device)
        elif args.model_type == 'transformer':
            model = Informer(enc_in=2*k, dec_in=2*k, c_out=2*k, out_len=pred_len, attn="full").to(device)
        
        criterion = NMSELoss().to(device)    
        error_nmse = 0
        num = 0
        epoch_val_loss = []
        
        # ============ Deep Learning Models =============== #
        if args.model_type in ['lstm', 'llm4cp', 'transformer']:
            if args.resume:
                checkpoint_path = f"{args.resume}{args.model_type}_{dataset_name}_fullshot/checkpoint-149.pth"
                
                if os.path.isfile(checkpoint_path):
                    print(f"=> Loading checkpoint: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    # 兼容不同保存格式
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    print(f"=> [Warning] Checkpoint not found: {checkpoint_path}")
                    print("=> Testing with random weights (Results will be invalid)!")

            # ============ Validating =============== #
            model.eval()
            with torch.no_grad():
                for iteration, (samples, _, _) in enumerate(dataset_test, 1):
                    # samples: [B, T, K, U]
                    B, T, K, U = samples.shape
                    
                    # [Logic]: 时域切分
                    prev_len = int(T / 2)
                    pred_len = int(T / 2)
                    label_len = prev_len // 2 
                    
                    # 根据你的原始代码逻辑: samples[:, int(pred_len):, : ] 是 prev_data
                    prev_data, pred_data = samples[:, int(pred_len):, : ], samples[:, :int(pred_len), :]

                    # Permute: (B, T, K, U) -> (B, U, T, K)
                    # 目的: 将 T 放在序列维度, K 放在特征维度
                    prev_data = prev_data.permute(0, 3, 1, 2)
                    pred_data = pred_data.permute(0, 3, 1, 2)

                    # 转实数 (Last dim * 2)
                    prev_data = LoadBatch_ofdm_2(prev_data).to(device)
                    pred_data = LoadBatch_ofdm_2(pred_data).to(device)

                    # Rearrange: 合并 Batch 和 U -> (Batch*U, T, K*2)
                    prev_data = rearrange(prev_data, 'b u t k -> (b u) t k')
                    pred_data = rearrange(pred_data, 'b u t k -> (b u) t k')
                    
                    # Inference
                    if args.model_type in ['lstm']:
                        out = model(prev_data, pred_len, device)
                    elif args.model_type == 'llm4cp':
                        out = model(prev_data, None, None, None)
                    elif args.model_type in ['transformer']:
                        encoder_input = prev_data
                        dec_inp = torch.zeros_like(encoder_input[:, -pred_len:, :]).to(device)
                        decoder_input = torch.cat([encoder_input[:, prev_len - label_len:prev_len, :], dec_inp],
                                                    dim=1)
                        out = model(encoder_input, decoder_input)

                    
                    loss = criterion(out, pred_data)
                    epoch_val_loss.append(loss.item())

                    y_pred = out.reshape(-1,1).reshape(B,-1).detach().cpu().numpy()  
                    y_target = pred_data.reshape(-1,1).reshape(B,-1).detach().cpu().numpy()

                    # 计算 Linear NMSE
                    batch_nmse = np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
                    error_nmse += batch_nmse
                    num += B
                    
                    if iteration % 20 == 0:
                        print(f"Batch {iteration}, NMSE Accum: {error_nmse:.2f}")

        # ============ Traditional / Other Methods =============== #
        elif args.model_type in ['pad']:
            print(f"Running PAD for {dataset_name}...")
            for iteration, (samples, _, _) in enumerate(dataset_test, 1):
                B, T, K, U = samples.shape
                for idx in range(B):
                    pred_len = int(T / 2)
                    prev_data, pred_data = samples[idx, int(pred_len):, :, :], samples[idx, :int(pred_len), :, :]
                    
                    # PAD specific rearrange: t k u -> k t u
                    prev_data = rearrange(prev_data, 't k u -> k t u', k=K)
                    pred_data = rearrange(pred_data, 't k u -> k t u', k=K)

                    p = 4 if T == 16 else 6
                    out = PAD3(prev_data, p=p, startidx=pred_len, subcarriernum=K, Nr=1, Nt=u, pre_len=pred_len)
                    
                    out = LoadBatch_ofdm_1(out)
                    pred = LoadBatch_ofdm_1(pred_data)
                    
                    # PAD loss accumulation (Note: this is sum of NMSEs effectively if criterion is not mean reduced)
                    loss = criterion(out, pred)
                    error_nmse += loss.item()
                
                num += B

        # ============ End of Single Dataset Loop ============ #
        if num > 0:
            # 1. 计算当前数据集的平均【线性】NMSE
            nmse_linear = error_nmse / num
            
            # 2. 转换为 dB 用于展示
            nmse_db = 10 * np.log10(np.clip(nmse_linear, 1e-10, None))
            
            v_loss = np.nanmean(np.array(epoch_val_loss)) if len(epoch_val_loss) > 0 else 0
            
            print(f'Dataset: {dataset_name}, Val Loss: {v_loss:.7f}, NMSE (Lin): {nmse_linear:.7f}, NMSE (dB): {nmse_db:.4f}')   
            
            # 3. 写入文件
            with open(result_file_path, 'a') as f:
                f.write(f"{dataset_name}, {nmse_db:.4f} dB\n")
            
            # 4. 收集线性值，用于最后计算总平均
            all_nmse_linear.append(nmse_linear)
        else:
            print(f"No samples processed for {dataset_name}")

    # ============ 所有数据集测试结束，计算总平均 ============ #
    if len(all_nmse_linear) > 0:
        # [Critical]: 先对线性误差求平均，再转 dB，这样更科学
        avg_linear = np.mean(all_nmse_linear)
        avg_db = 10 * np.log10(np.clip(avg_linear, 1e-10, None))
        
        print(f"\n{'='*20} All Tasks Finished {'='*20}")
        print(f"Datasets Evaluated: {dataset_list}")
        print(f"Average Linear NMSE: {avg_linear:.7f}")
        print(f"Average NMSE (dB):   {avg_db:.4f} dB")
        
        # 保存平均值到文件
        with open(result_file_path, 'a') as f:
            f.write(f"Avg, {avg_db:.4f} dB\n")
            
    print(f"Full results saved to: {result_file_path}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)