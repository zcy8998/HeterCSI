# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import pdb
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pdb.set_trace = lambda *args, **kwargs: None

import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# from torchao.quantization.linear_activation_quantized_tensor import \
#     LinearActivationQuantizedTensor

import models.CSIGPT as CSIGPT
import timm_utils.optim.optim_factory as optim_factory
import util.misc as misc
from engine_pretrain import train_one_epoch_3mask
from util.data import data_load_main
from util.metrics import NMSELoss
from util.misc import NativeScalerWithGradNormCount as NativeScaler

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16_128', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=96, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='images input size')
    parser.add_argument('--mask_type', default='random', choices=['random', 'temporal', 'freq', 'all'],
                        help='Masking strategy for the input data')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--spatial_mask', action='store_true', default=False,
                        help='whether to mask all channels of a spatial location. Only for indp c model')
    parser.add_argument('--cls_token', action='store_true', default=False,
                        help='Whether to add class token') 
    parser.add_argument('--depth', default=12, type=int,
                        help='the depth of the transformer encoder')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_path', default='', type=str,
                        help='Train.csv path')
    parser.add_argument('--shuffle_type', default='global',
                        choices=['gloabl', 'bucket'],
                        help='Whether to use fmow rgb, sentinel, or other dataset.')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_dir', default='', type=str,
                        help='directory containing multiple checkpoints to evaluate')
    parser.add_argument('--resume_different_size', default='',
                        help='continue to pretrain for different size')
    parser.add_argument('--wandb', type=str, default=None,
                        help="Wandb project name, eg: sentinel_pretrain")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=os.getenv('LOCAL_RANK', 0), type=int)  # prev default was -1
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dataset', default="D1", type=str,
                        help='dataset used to train')
    parser.add_argument('--max_length', default=1024, type=int,
                        help='max seq len after patching')
    parser.add_argument('--data_dir', default=None, type=str,
                        help='the data dir used to train')
    parser.add_argument('--data_num', default=1.0, type=float,
                        help='the amount of data used to finetune')

    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if not args.distributed:
        args.gpu = "cuda:0"
    device = torch.device(args.gpu)
    # device = torch.device("cpu")

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    global_rank = misc.get_rank()
    
    if global_rank == 0:
        dataset_val_all = data_load_main(args, dataset_type='test', test_type='normal') # 加载数据

    checkpoints = [os.path.join(args.resume, f"checkpoint-{i}.pth") for i in range(args.epochs)]
    
    # 初始化每个掩码类型的统计容器
    if args.mask_type == 'all':
        mask_list = {'random': 0.85, 'temporal': 0.5, 'freq': 0.5}
    else:
        mask_list = {args.mask_type: args.mask_ratio}

    for idx, resume_path in enumerate(checkpoints):
        # instantiate a fresh model for each checkpoint
        print(f"\n=== Evaluating checkpoint: {resume_path or 'NONE'} ===")
        model = CSIGPT.__dict__[args.model](
            cls_embed=args.cls_token,
            device=device
        )
        model.to(device)
        model_without_ddp = model

        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
        loss_scaler = NativeScaler()

        # set resume path for loading
        args.resume = resume_path
        pdb.set_trace()
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        model.eval()

        if global_rank == 0:
            total = sum([param.nelement() for param in model.parameters()])
            print("Number of parameter: %.5fM" % (total / 1e6))
            total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

            model.eval()
            all_task_results = {}
            all_task_nmse = []

            for task_idx, (mask_type, mask_ratio) in enumerate(mask_list.items()):
                nmse_list = []
                for index, dataset_test in enumerate(dataset_val_all):
                    dataset_name = dataset_test.dataset.get_dataset_name()
                    print(f"Start test {dataset_name}.")
                    with torch.no_grad():
                        error_nmse = 0
                        num = 0
                        epoch_val_loss = []
                        total_inference_time = 0.0
                        num_batches = 0

                        # 重置 GPU 缓存（确保干净起点）
                        if torch.cuda.is_available() and device.type == 'cuda':
                            torch.cuda.reset_peak_memory_stats(device)
                            torch.cuda.empty_cache()

                        for iteration, (samples, token_length, input_size) in enumerate(dataset_test, 1):
                            optimizer.zero_grad()
                            samples = samples.to(device)
                            token_length = token_length.to(device)

                            # 同步 GPU（确保时间准确）
                            if device.type == 'cuda':
                                torch.cuda.synchronize()

                            start_time = time.perf_counter()
                            loss, pred, mask = model(samples, token_length, input_size,
                                                    mask_ratio=mask_ratio, mask_strategy=mask_type)
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                            end_time = time.perf_counter()

                            total_inference_time += (end_time - start_time)
                            num_batches += 1

                            # --- NMSE 计算（保持不变）---
                            N, L, D = samples.shape
                            col_indices = torch.arange(L, device=samples.device).expand(N, L)
                            mask_in_length = col_indices < token_length[:, None]
                            bool_mask = mask.bool()
                            mask_nmse = bool_mask & mask_in_length

                            N = pred.shape[0]
                            y_pred = pred[mask_nmse == 1].reshape(-1, 1).reshape(N, -1).detach().cpu().numpy()
                            y_target = samples[mask_nmse == 1].reshape(-1, 1).reshape(N, -1).detach().cpu().numpy()

                            error_nmse += np.sum(
                                np.mean(np.abs(y_target - y_pred) ** 2, axis=1) /
                                np.mean(np.abs(y_target) ** 2, axis=1)
                            )
                            num += y_pred.shape[0]
                            epoch_val_loss.append(loss.item())

                        nmse = error_nmse / num
                        v_loss = np.nanmean(np.array(epoch_val_loss))
                        avg_inference_time = total_inference_time / num_batches if num_batches > 0 else 0.0


                        # 打印结果
                        log_str = (f'dataset_name: {dataset_name}, '
                                f'Validation loss: {v_loss:.7f}, '
                                f'NMSE: {nmse:.7f}, '
                                f'Avg Inference Time per Batch: {avg_inference_time * 1000:.3f} ms')
                        print(log_str)

                        nmse_list.append(nmse)

                print(f"Task type is {mask_type}, Average NMSE for all datasets: {np.mean(nmse_list):.7f}")
                all_task_nmse.append(np.mean(nmse_list))

                # 存储当前任务结果：包含每个数据集的NMSE和平均NMSE
                all_task_results[mask_type] = {
                    "avg_nmse": np.mean(nmse_list)  # 该任务所有数据集的平均NMSE
                }
        
            all_task_results["all"] = {
                "avg_nmse": np.mean(all_task_nmse)  # 该任务所有数据集的平均NMSE
            }

            # 构建保存路径
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
            save_path = os.path.join(output_dir, f"nmse_{idx}.pth")

            # 保存结果字典
            torch.save(all_task_results, save_path)
            print(f"Results saved to {save_path}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)