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
    # H: B,T,K,mul     [tensor complex]
    # out:B,T,K,mul*2  [tensor real]
    B, T, K, mul = H.shape
    H_real = np.zeros([B, T, K, mul, 2])
    H_real[:, :, :, :, 0] = H.real
    H_real[:, :, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, K, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real

def LoadBatch_ofdm_2_tensor(H):
    # H: B, T, K, mul [Complex Tensor]
    # out: B, T, K, mul*2 [Real Tensor]
    
    # 确保是 Tensor
    if not torch.is_tensor(H):
        H = torch.tensor(H)
        
    # 如果是复数 Tensor，直接使用 PyTorch 高效 API
    if H.is_complex():
        # view_as_real 将复数拆分为 (..., 2) 即 [real, imag]
        # flatten 将最后两维合并: mul, 2 -> mul*2
        return torch.view_as_real(H).flatten(start_dim=-2).float()
    else:
        # 如果输入不是复数类型（或者是已经分开的float），按需处理
        # 这里假设输入必定是复数数据
        return H.float()

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

    prev_len = int(t / 2)
    pred_len = int(t / 2)
    label_len = prev_len // 2 

    dataset_train = data_load_baseline(args, dataset_type='train', data_num=args.data_num) # 加载数据
    dataset_val = data_load_baseline(args, dataset_type='val') # 加载数据


    if args.model_type == 'lstm':
        model = LSTM(features=2*k, input_size=2*k, hidden_size=4*k, num_layers=4).to(device)
    elif args.model_type == 'llm4cp':
        model = LLM4CP(pred_len=pred_len, prev_len=prev_len, K=k, UQh=1, UQv=1, BQh=1, BQv=1).to(device)
    elif args.model_type == 'transformer':
        model = Informer(enc_in=2*k, dec_in=2*k, c_out=2*k, out_len=pred_len, attn="full").to(device)

    
    print("Model = %s" % str(model))

    # following timm: set wd as 0 for bias and norm layers
    criterion = NMSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    # misc.load_model_different_size(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    if args.resume:       
        model.load_state_dict(torch.load(args.resume)) 

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
        
    print(f'Start training...{model}')
    for epoch in range(args.epochs):
        start_time = time.time()  # 记录当前时间
        epoch_train_loss, epoch_val_loss = [], []
        # ============Epoch Train=============== #
        model.train()

        for iteration, (samples, _, _) in enumerate(dataset_train, 1):
            B, T, K, U = samples.shape
            pred_len = int(T / 2)
            prev_data, pred_data = samples[:, int(pred_len):, : ], samples[:, :int(pred_len), :]
            pdb.set_trace()
            prev_data = prev_data.permute(0, 3, 1, 2)
            pred_data = pred_data.permute(0, 3, 1, 2)

            prev_data = prev_data.to(device, non_blocking=True)
            pred_data = pred_data.to(device, non_blocking=True)

            prev_data = LoadBatch_ofdm_2_tensor(prev_data)
            pred_data = LoadBatch_ofdm_2_tensor(pred_data)

            prev_data = rearrange(prev_data, 'b u t k -> (b u) t k')
            pred_data = rearrange(pred_data, 'b u t k -> (b u) t k')
            pdb.set_trace()
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

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        epoch_time = time.time() - start_time  # 计算训练时间
        print('Epoch: {}/{} training loss: {:.7f}, time: {:.2f}s'.format(epoch+1, args.epochs, t_loss, epoch_time)) 

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, (samples, _, _) in enumerate(dataset_val, 1):
                B, T, K, U = samples.shape
                pred_len = int(T / 2)
                prev_data, pred_data = samples[:, int(pred_len):, : ], samples[:, :int(pred_len), :]

                prev_data = prev_data.permute(0, 3, 1, 2)
                pred_data = pred_data.permute(0, 3, 1, 2)

                prev_data = LoadBatch_ofdm_2(prev_data).to(device)
                pred_data = LoadBatch_ofdm_2(pred_data).to(device)

                prev_data = rearrange(prev_data, 'b u t k -> (b u) t k')
                pred_data = rearrange(pred_data, 'b u t k -> (b u) t k')
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
                epoch_val_loss.append(loss.item())  # save all losses into a vector for one epoch
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
