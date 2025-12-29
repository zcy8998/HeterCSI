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
    # task_type = 'temporal'
    # model_path = {
    #     'lstm': f'/mnt/4T/2/zcy/CSIGPT/baseline_temporal_lstm_D1/checkpoint-149.pth',
    #     'llm4cp': f'/mnt/4T/2/zcy/CSIGPT/baseline_temporal_llm4cp_D1/checkpoint-149.pth',
    # }
    NMSE = []

    path = os.path.join(args.data_dir, args.dataset, "test_data.mat")
    with h5py.File(path, 'r') as f:
        dset = f[f'H_test']
        U, K, T, _ = dset.shape
        print("U, K, T:", U, K, T)
        t, k, u = T, K, U

    pred_len = int(t / 2)
    
    dataset_test = data_load_baseline(args, dataset_type='test') # 加载数据

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
    if args.model_type in ['lstm', 'llm4cp', 'transformer']:
        print("Model = %s" % str(model))

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
                B, T, K, U = samples.shape
                prev_len = int(T / 2)
                pred_len = int(T / 2)
                label_len = prev_len // 2 
                prev_data, pred_data = samples[:, int(pred_len):, : ], samples[:, :int(pred_len), :]

                prev_data = prev_data.permute(0, 3, 1, 2)
                pred_data = pred_data.permute(0, 3, 1, 2)

                prev_data = LoadBatch_ofdm_2(prev_data).to(device)
                pred_data = LoadBatch_ofdm_2(pred_data).to(device)

                prev_data = rearrange(prev_data, 'b u t k -> (b u) t k')
                pred_data = rearrange(pred_data, 'b u t k -> (b u) t k')
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

                y_pred = out.reshape(-1,1).reshape(B,-1).detach().cpu().numpy()  # [Batch_size, 样本点数目]
                y_target = pred_data.reshape(-1,1).reshape(B,-1).detach().cpu().numpy()

                error_nmse += np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
                num += B
                print(error_nmse, B)

            nmse = error_nmse / num
            v_loss = np.nanmean(np.array(epoch_val_loss))
            nmse_list.append(nmse)
            print(f'dataset_name: {args.dataset}, Validation loss: {v_loss:.7f}, NMSE: {nmse:.7f}')   
    elif args.model_type in ['pad']:
        for iteration, (samples, _, _) in enumerate(dataset_test, 1):
            B, T, K, U = samples.shape
            for idx in range(B):
                pred_len = int(T / 2)
                prev_data, pred_data = samples[idx, int(pred_len):, :, :], samples[idx, :int(pred_len), :, :]
                prev_data = rearrange(prev_data, 't k u -> k t u', k=K)
                pred_data = rearrange(pred_data, 't k u -> k t u', k=K)

                p = 4 if T == 16 else 6
                out = PAD3(prev_data, p=p, startidx=pred_len, subcarriernum=K, Nr=1, Nt=u, pre_len=pred_len)
                
                pdb.set_trace()
                out = LoadBatch_ofdm_1(out)
                pred = LoadBatch_ofdm_1(pred_data)
                loss = criterion(out, pred)
                
                error_nmse += loss
            
            num += B

        nmse = error_nmse / num
        v_loss = np.nanmean(np.array(epoch_val_loss))
        print(f'dataset_name: {args.dataset}, Validation loss: {v_loss:.7f}, NMSE: {nmse:.7f}')   



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
