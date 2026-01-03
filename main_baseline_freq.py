import argparse
import datetime
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TORCH_NUM_THREADS"] = "8"

import h5py
import pdb
import time
from pathlib import Path

import numpy as np
pdb.set_trace = lambda *args, **kwargs: None

import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim

# 请确保这些引用路径在你的项目中是正确的
from models.baseline.model import *
from models.baseline.LLM4CP import Model as LLM4CP
from util.data import *
from util.misc import NativeScalerWithGradNormCount as NativeScaler

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model_type', default='lstm', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--seed', default=0, type=int)

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--dataset', default="D1", type=str,
                        help='dataset used to train')
    parser.add_argument('--data_num', default=1.0, type=float,
                        help='data num used to finetune')
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


def LoadBatch_ofdm_2(H):
    # H: B, Seq_Len, Feature_Dim, mul (complex)
    # out: B, Seq_Len, Feature_Dim, mul*2 (real)
    # [Note]: 这个函数会将最后一个维度(mul)展开为实部虚部。
    # 在频域预测中，传入的shape将是 (B, U, K_split, T)，mul对应T
    B, T, K, mul = H.shape
    H_real = np.zeros([B, T, K, mul, 2])
    H_real[:, :, :, :, 0] = H.real
    H_real[:, :, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, K, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real

# 修改后的函数，支持 GPU Tensor 操作，移除 Numpy 依赖
def LoadBatch_ofdm_2_tensor(H):
    # H: B, T, K, mul (complex via last dim usually? 
    # based on your code: H_real[:, :, :, :, 0] = H.real)
    # 你的原始代码中 H 似乎是一个 复数 numpy 数组或者包含复数的 Tensor？
    # 如果 H 是 PyTorch 的 Complex Tensor (complex64/128):
    
    if not torch.is_tensor(H):
        H = torch.tensor(H)

    if not H.is_contiguous():
        H = H.contiguous()
    # 假设输入已经是 (B, T, K, mul) 且在 GPU 上
    # 如果输入本身就是 Complex Tensor
    if H.is_complex():
        # Stack real and imag in a new last dimension
        H_real = torch.stack([H.real, H.imag], dim=-1) # (B, T, K, mul, 2)
    else:
        # 如果输入本来就是分开的或者是其他格式，需要根据上游数据格式调整
        # 这里假设它是复数 tensor
        pass

    # Flatten the last two dimensions: mul * 2
    B, T, K, mul = H.shape
    # view/reshape 是零拷贝操作，极快
    H_out = torch.view_as_real(H).view(B, T, K, -1) 
    
    return H_out.float()

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

    path = os.path.join(args.data_dir, args.dataset, "test_data.mat")
    with h5py.File(path, 'r') as f:
        dset = f[f'H_test']
        U, K, T, _ = dset.shape
        print("U, K, T:", U, K, T)
        t, k, u = T, K, U

    # [Modified]: 预测维度变为 K (Subcarriers)，而非 T
    # 我们假设在频域上进行预测，例如根据一部分子载波预测另一部分，或者Masked Modeling
    prev_len = int(k / 2) 
    pred_len = int(k / 2)
    label_len = prev_len // 2 

    dataset_train = data_load_baseline(args, dataset_type='train', data_num=args.data_num) 
    dataset_val = data_load_baseline(args, dataset_type='val') 

    # [Modified]: 模型输入/特征维度调整
    # 频域预测中：
    # 序列长度(Sequence Length) = K (子载波数量)
    # 特征维度(Input Size) = 2 * t (每个子载波位置包含所有时间步的实部和虚部)
    
    input_feat_dim = 2 * t # [Modified] 以前是 2*k

    if args.model_type == 'lstm':
        # input_size 变为 2*t
        model = LSTM(features=input_feat_dim, input_size=input_feat_dim, hidden_size=2*input_feat_dim, num_layers=4).to(device)
    elif args.model_type == 'llm4cp':
        # [Modified] LLM4CP 通常需要指定序列长度参数。
        # 如果 LLM4CP 的 K 参数代表序列长度，这里应该传 k (但如果它内部用K做特征维度，则需传入 t，视具体实现而定)
        # 假设参数 K 代表 "Feature Dimension" 或者 "Subcarrier dimension"，这里我们需要反转概念：
        # 现在的"Sequence"是K，"Feature"是T。
        # 此处假设 LLM4CP 的 K 参数指的是 Input Feature Size 的一半 (Complex数量)
        model = LLM4CP(pred_len=pred_len, prev_len=prev_len, K=t, UQh=1, UQv=1, BQh=1, BQv=1).to(device)
    elif args.model_type == 'transformer':
        # enc_in, dec_in, c_out 变为 2*t
        model = Informer(enc_in=input_feat_dim, dec_in=input_feat_dim, c_out=input_feat_dim, out_len=pred_len, attn="full").to(device)
    
    print("Model = %s" % str(model))

    criterion = NMSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    if args.resume:        
        model.load_state_dict(torch.load(args.resume)) 

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
        
    print(f'Start training...{model}')
    for epoch in range(args.epochs):
        start_time = time.time() 
        epoch_train_loss, epoch_val_loss = [], []
        # ============Epoch Train=============== #
        model.train()

        for iteration, (samples, _, _) in enumerate(dataset_train, 1):
            # samples shape: B, T, K, U
            B, T, K, U = samples.shape
            
            # [Modified]: 切分维度变为 K (index 2)
            pred_len = int(K / 2) # 更新 pred_len 以防 batch 中 K 变化（通常不变）
            
            # 原始数据通常是 [B, T, K, U]
            # 频域预测：根据前一半K预测后一半K
            # prev_data (Input): samples[:, :, int(pred_len):, :]  <-- 这里根据你的任务逻辑调整
            # 假设任务是：已知 K/2 -> 预测 K/2
            # 通常 prev_data 是输入， pred_data 是标签
            
            # [Note]: 这里根据原代码逻辑：input是后半段，label是前半段？原代码逻辑比较特殊。
            # 如果你是要做常规预测（前 -> 后），请反过来。
            # 这里保留原代码的切分逻辑，只是换了维度：
            prev_data, pred_data = samples[:, :, int(pred_len):, : ], samples[:, :, :int(pred_len), :] 

            prev_data = prev_data.to(device, non_blocking=True) 
            pred_data = pred_data.to(device, non_blocking=True)
            
            # [Modified]: Permute 维度置换
            # 目标: (Batch, User, Subcarrier, Time) -> (B, U, K, T)
            # 这样 LoadBatch 处理后，最后一维 T 会变成特征 T*2
            prev_data = prev_data.permute(0, 3, 2, 1) # [Modified] Old: (0, 3, 1, 2)
            pred_data = pred_data.permute(0, 3, 2, 1) # [Modified] Old: (0, 3, 1, 2)

            # LoadBatch 会把最后一维当做复数展开。
            # 输入 (B, U, K_part, T) -> 输出 (B, U, K_part, T*2)
            prev_data = LoadBatch_ofdm_2_tensor(prev_data)
            pred_data = LoadBatch_ofdm_2_tensor(pred_data).to(device)

            # [Modified]: Rearrange
            # 这里的语义变成了：Batch=b*u, Sequence=k, Feature=t
            # 对应的 LoadBatch 后的 tensor 维度是 [B, U, K, T*2]
            prev_data = rearrange(prev_data, 'b u k t -> (b u) k t') # [Modified] t is feature
            pred_data = rearrange(pred_data, 'b u k t -> (b u) k t') # [Modified] t is feature
            
            optimizer.zero_grad()  # fixed
            if args.model_type in ['lstm']:
                out = model(prev_data, pred_len, device)
            elif args.model_type in ['llm4cp']:
                out = model(prev_data, None, None, None)
            elif args.model_type in ['transformer']:
                encoder_input = prev_data
                dec_inp = torch.zeros_like(encoder_input[:, -pred_len:, :]).to(device)
                decoder_input = torch.cat([encoder_input[:, prev_len - label_len:prev_len, :], dec_inp],
                                          dim=1)
                out = model(encoder_input, decoder_input)

            loss = criterion(out, pred_data)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()
            optimizer.step()

        t_loss = np.nanmean(np.array(epoch_train_loss)) 
        epoch_time = time.time() - start_time 
        print('Epoch: {}/{} training loss: {:.7f}, time: {:.2f}s'.format(epoch+1, args.epochs, t_loss, epoch_time)) 

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, (samples, _, _) in enumerate(dataset_val, 1):
                B, T, K, U = samples.shape
                pred_len = int(K / 2) # [Modified] K

                # [Modified] Slicing on K (dim 2)
                prev_data, pred_data = samples[:, :, int(pred_len):, : ], samples[:, :, :int(pred_len), :]

                # [Modified] Permute to put T last
                prev_data = prev_data.permute(0, 3, 2, 1) 
                pred_data = pred_data.permute(0, 3, 2, 1)

                prev_data = LoadBatch_ofdm_2(prev_data).to(device)
                pred_data = LoadBatch_ofdm_2(pred_data).to(device)

                # [Modified] Rearrange k as seq, t as feat
                prev_data = rearrange(prev_data, 'b u k t -> (b u) k t')
                pred_data = rearrange(pred_data, 'b u k t -> (b u) k t')

                optimizer.zero_grad()  # fixed
                if args.model_type in ['lstm']:
                    out = model(prev_data, pred_len, device)
                elif args.model_type in ['llm4cp']:
                    out = model(prev_data, None, None, None)
                elif args.model_type in ['transformer']:
                    encoder_input = prev_data
                    dec_inp = torch.zeros_like(encoder_input[:, -pred_len:, :]).to(device)
                    decoder_input = torch.cat([encoder_input[:, prev_len - label_len:prev_len, :], dec_inp],
                                                dim=1)
                    out = model(encoder_input, decoder_input)

                loss = criterion(out, pred_data) 
                epoch_val_loss.append(loss.item()) 
            v_loss = np.nanmean(np.array(epoch_val_loss))
            print('validate loss: {:.7f}'.format(v_loss))
            if ((epoch + 1) % 50) == 0:
                save_best_checkpoint(model, os.path.join(args.output_dir,f"checkpoint-{epoch}.pth"))
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)