import argparse
import datetime
import os
import pdb
import time
from pathlib import Path
import h5py # 补上 missing import

import numpy as np

pdb.set_trace = lambda *args, **kwargs: None

import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim

from models.baseline.model import *
from models.baseline.LLM4CP import Model as LLM4CP
from models.baseline.PAD import PAD3
from util.data import *
from util.misc import NativeScalerWithGradNormCount as NativeScaler

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model_type', default='lstm', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task_type', default='temporal', type=str, help='Name of task to test')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--dataset', default="D1", type=str,
                        help='dataset used to train')
    parser.add_argument('--data_dir', default=None, type=str,
                        help='the data dir used to train')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

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


def save_best_checkpoint(model, save_path):  # save model function
    model_out_path = save_path
    torch.save(model.state_dict(), model_out_path)


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
    # [Note]: 此函数只是将最后一个复数维度展开，不改变 T 和 K 的位置
    # 在频域预测中，传入的 T 其实是逻辑上的 K (Sequence)，传入的 K 其实是逻辑上的 T (Feature)
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
  
    # [Modified]: 1. 准备结果文件
    result_file_path = os.path.join(args.log_dir, f"{args.model_type}_evaluation_results_freq.txt")
    with open(result_file_path, 'w') as f:
        f.write(f"Task Type: {args.model_type} (Frequency Prediction)\n")
        f.write("Dataset, NMSE (dB)\n")

    # [Modified]: 2. 解析数据集列表 (例如 "Cost2100,CDL_A")
    dataset_list = args.dataset.split(',')

    all_nmse_linear = []

    # ============ 循环测试每个数据集 ============ #
    for dataset_name in dataset_list:
        dataset_name = dataset_name.strip()
        print(f"\n{'='*20} Testing Dataset: {dataset_name} {'='*20}")

        # 更新 args.dataset 以便 data_load_baseline 加载正确的数据
        args.dataset = dataset_name

        # [Modified]: 3. 读取当前数据集的维度 (不同数据集 K, T 可能不同)
        path = os.path.join(args.data_dir, dataset_name, "test_data.mat")
        try:
            with h5py.File(path, 'r') as f:
                dset = f[f'H_test']
                U, K, T, _ = dset.shape
                print("U, K, T:", U, K, T)
                t, k, u = T, K, U
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue

        # 预测维度设置
        pred_len = int(k / 2) 
        dataset_test = data_load_baseline(args, dataset_type='test') # 加载数据
        input_feat_dim = 2 * t

        # [Modified]: 4. 重新初始化模型 (因为 input_feat_dim 和 K 可能随数据集变化)
        if args.model_type == 'lstm':
            model = LSTM(features=input_feat_dim, input_size=input_feat_dim, hidden_size=2*input_feat_dim, num_layers=4).to(device)
        elif args.model_type == 'llm4cp':
            model = LLM4CP(pred_len=pred_len, prev_len=pred_len, K=t, UQh=1, UQv=1, BQh=1, BQv=1).to(device)
        elif args.model_type == 'transformer':
            model = Informer(enc_in=input_feat_dim, dec_in=input_feat_dim, c_out=input_feat_dim, out_len=pred_len, attn="full").to(device)
        
        criterion = NMSELoss().to(device)    
        error_nmse = 0
        num = 0
        epoch_val_loss = []
        
        if args.model_type in ['lstm', 'llm4cp', 'transformer']:
            # [Modified]: 5. 动态构建 Checkpoint 路径
            # 格式: args.resume + "_" + model_type + "_" + dataset + "_fullshot_freq/checkpoint-149.pth"
            if args.resume:
                checkpoint_path = f"{args.resume}{args.model_type}_{dataset_name}_fullshot_freq/checkpoint-149.pth"
                
                if os.path.isfile(checkpoint_path):
                    print(f"=> Loading checkpoint: {checkpoint_path}")
                    # 使用 map_location 确保加载到正确的设备
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    # 如果 checkpoint 保存的是 'state_dict' 键，则加载它，否则直接加载
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    print(f"=> [Warning] No checkpoint found at '{checkpoint_path}'. Testing with random weights!")
            
            # ============ Validating =============== #
            model.eval()
            with torch.no_grad():
                for iteration, (samples, _, _) in enumerate(dataset_test, 1):
                    # samples: [B, T, K, U]
                    B, T, K, U = samples.shape
                    
                    prev_len = int(K / 2)
                    pred_len = int(K / 2)
                    label_len = prev_len // 2 
                    
                    # Slicing on dim 2 (K) - Frequency prediction
                    prev_data, pred_data = samples[:, :, int(pred_len):, : ], samples[:, :, :int(pred_len), :]

                    # Permute: (B, U, K, T)
                    prev_data = prev_data.permute(0, 3, 2, 1) 
                    pred_data = pred_data.permute(0, 3, 2, 1)

                    prev_data = LoadBatch_ofdm_2(prev_data).to(device)
                    pred_data = LoadBatch_ofdm_2(pred_data).to(device)

                    # Rearrange: Batch*U as sequence, K as length, T as feature
                    prev_data = rearrange(prev_data, 'b u k t -> (b u) k t')
                    pred_data = rearrange(pred_data, 'b u k t -> (b u) k t')
                    
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

                    error_nmse += np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
                    num += B
                    
                    # 减少打印频率，避免日志过多
                    if iteration % 10 == 0:
                        print(f"Dataset {dataset_name} | Batch {iteration} | Cur NMSE sum: {error_nmse:.2f}")

                # 计算最终指标
                if num > 0:
                    nmse = error_nmse / num
                    nmse_db = 10 * np.log10(np.clip(nmse, 1e-10, None))
                    v_loss = np.nanmean(np.array(epoch_val_loss))
                    print(f'dataset_name: {dataset_name}, Validation loss: {v_loss:.7f}, NMSE: {nmse:.7f}, NMSE (dB): {nmse_db:.4f}')   
                    
                    # [Modified]: 6. 将结果写入文件
                    with open(result_file_path, 'a') as f:
                        f.write(f"{dataset_name}, {nmse_db:.4f} dB\n")
                    all_nmse_linear.append(nmse)
                else:
                    print(f"Dataset {dataset_name} has no samples.")

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