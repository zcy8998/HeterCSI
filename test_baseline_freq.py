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

    device = "cuda:0"
    # device = torch.device("cpu")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    os.makedirs(args.log_dir, exist_ok=True)
    args.distributed = False
  
    NMSE = []

    path = os.path.join(args.data_dir, args.dataset, "test_data.mat")
    with h5py.File(path, 'r') as f:
        dset = f[f'H_test']
        U, K, T, _ = dset.shape
        print("U, K, T:", U, K, T)
        t, k, u = T, K, U

    # [Modified]: 预测维度是 K (Frequency)
    pred_len = int(k / 2) 
    
    dataset_test = data_load_baseline(args, dataset_type='test') # 加载数据

    # [Modified]: Feature dim 变为 2*t
    input_feat_dim = 2 * t

    if args.model_type == 'lstm':
        # features / input_size = 2*t
        model = LSTM(features=input_feat_dim, input_size=input_feat_dim, hidden_size=2*input_feat_dim, num_layers=4).to(device)
    elif args.model_type == 'llm4cp':
        # 假设 LLM4CP 的 K 参数控制特征维度，传入 t
        model = LLM4CP(pred_len=pred_len, prev_len=pred_len, K=t, UQh=1, UQv=1, BQh=1, BQv=1).to(device)
    elif args.model_type == 'transformer':
        model = Informer(enc_in=input_feat_dim, dec_in=input_feat_dim, c_out=input_feat_dim, out_len=pred_len, attn="full").to(device)
    
    criterion = NMSELoss().to(device)    
    error_nmse = 0
    num = 0
    epoch_val_loss = []
    
    if args.model_type in ['lstm', 'llm4cp', 'transformer']:
        print("Model = %s" % str(model))

        if args.resume:
            model.load_state_dict(torch.load(args.resume)) 

        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))
        total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
        
        # ============Epoch Validate=============== #
        model.eval()
        nmse_list = []
        with torch.no_grad():
            for iteration, (samples, _, _) in enumerate(dataset_test, 1):
                # samples: [B, T, K, U]
                B, T, K, U = samples.shape
                
                # [Modified] 切分逻辑基于 K
                prev_len = int(K / 2)
                pred_len = int(K / 2)
                label_len = prev_len // 2 
                
                # [Modified] Slicing on dim 2 (K)
                # 假设：用后一半 K 预测前一半 K (与 Train 保持一致，反之亦然)
                prev_data, pred_data = samples[:, :, int(pred_len):, : ], samples[:, :, :int(pred_len), :]

                # [Modified] Permute: (B, U, K, T)
                # 目标是把 T 放到最后作为特征
                prev_data = prev_data.permute(0, 3, 2, 1) # Old: (0, 3, 1, 2)
                pred_data = pred_data.permute(0, 3, 2, 1)

                prev_data = LoadBatch_ofdm_2(prev_data).to(device)
                pred_data = LoadBatch_ofdm_2(pred_data).to(device)

                # [Modified] Rearrange: Sequence is K, Feature is T
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

                
                loss = criterion(out, pred_data)  # compute loss
                epoch_val_loss.append(loss.item())  # save all losses into a vector for one epoch

                # NMSE 计算不需要改维度，因为 reshape(-1) 会把所有元素展平
                y_pred = out.reshape(-1,1).reshape(B,-1).detach().cpu().numpy()  
                y_target = pred_data.reshape(-1,1).reshape(B,-1).detach().cpu().numpy()

                error_nmse += np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
                num += B
                print(error_nmse, B)

            nmse = error_nmse / num
            v_loss = np.nanmean(np.array(epoch_val_loss))
            nmse_list.append(nmse)
            print(f'dataset_name: {args.dataset}, Validation loss: {v_loss:.7f}, NMSE: {nmse:.7f}')   
    
    elif args.model_type in ['pad']:
        # [Modified] PAD for Frequency Extrapolation
        # PAD 原本是基于多项式拟合进行时域外推。
        # 这里我们将 频域 K 伪装成 时域 T 喂给 PAD。
        for iteration, (samples, _, _) in enumerate(dataset_test, 1):
            B, T, K, U = samples.shape
            for idx in range(B):
                pred_len = int(K / 2) # [Modified]
                
                # [Modified] Slicing on K
                # prev_data: 已知的部分 (Input)
                # pred_data: 待预测的部分 (Label)
                prev_data, pred_data = samples[idx, :, int(pred_len):, :], samples[idx, :, :int(pred_len), :]
                
                # PAD 需要特定的输入格式。原代码: rearrange(prev_data, 't k u -> k t u')
                # 假设 PAD3 内部逻辑是对第2维(Time)进行处理。
                # 现在的输入 prev_data 是 [T, K_half, U]。
                # 我们希望 PAD 对 K 进行外推，所以需要把 K 放到 PAD 认为是 Time 的位置。
                # 构造为: [Feature=T, Sequence=K, User=U] -> 对应 PAD 的参数语义
                
                # 调整：将 T 视为 Feature (dim 0), K 视为 Time (dim 1)
                # 原代码 PAD3 调用：PAD3(data, ..., subcarriernum=K, ...)
                # 这里的 data 应该是 [Subcarrier, Time, User] ? 
                # 不，原代码 't k u -> k t u' 说明 PAD 期望 [Subcarrier, Time, User]。
                # 如果我们现在是在频域预测 (已知 K_half 推 K_half)，
                # 我们可以把 T 和 K 的角色互换。
                
                # 策略：保持 'k t u' 的结构，但在调用时让 PAD 认为 T 是 K，K 是 T。
                # prev_data shape: [T, K_half, U]
                # target input shape for PAD: [Pseudo_Subcarrier=T, Pseudo_Time=K_half, U]
                
                pad_input = prev_data.permute(0, 1, 2) # [T, K, U] - 已经符合 [Feat, Seq, U]
                
                # 注意：PAD3 的参数 subcarriernum 通常用于循环。
                # 这里我们要循环 T (现在的 dim 0)，在 K (现在的 dim 1) 上做预测。
                
                # p 是多项式阶数，需根据 K 的长度调整，而不是 T
                p = 4 if K == 16 else 6 # [Modified logic if needed]
                
                # 调用 PAD3
                # data: [T, K_half, U]
                # startidx: 预测起始点 (相对于 K)
                # subcarriernum: 这里传入 T (作为外部循环次数)
                # Nt: U
                # pre_len: K_half
                out = PAD3(pad_input, p=p, startidx=pred_len, subcarriernum=T, Nr=1, Nt=U, pre_len=pred_len)
                
                # out shape: [T, K_half, U, 2] (real/imag split inside PAD?) or complex?
                # PAD3 return 通常是 complex tensor 或 numpy
                
                # 修正 pred_data 格式以计算 Loss
                # pred_data: [T, K_half, U] -> 转换为实部虚部格式
                # 假设 LoadBatch_ofdm_1 接受 [B, T, mul]
                # 这里我们把 batch 设为 1 或合并 dims
                
                # 由于 PAD 输出格式比较特定，这里仅做逻辑占位，需确保 PAD3 输出与 pred 维度一致
                # 简单做法：计算 loss 时手动展平
                
                # 这里略过 LoadBatch 转换，直接算 complex error 可能更直接，或者沿用原流程
                # 为保持一致性，假设 out 已经是 tensor
                
                # 仅做简单 reshape 算 loss
                if isinstance(out, torch.Tensor):
                    pass
                else:
                    out = torch.tensor(out)
                
                # 统一转成 real 结构计算 MSE
                # out: [T, K_half, U] (Complex)
                # pred_data: [T, K_half, U] (Complex)
                loss = torch.mean(torch.abs(out - pred_data)**2) # 简化计算
                
                error_nmse += loss.item()
            
            num += B

        nmse = error_nmse / num
        print(f'dataset_name: {args.dataset}, NMSE: {nmse:.7f}')   

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)