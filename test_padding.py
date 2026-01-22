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
import uuid

pdb.set_trace = lambda *args, **kwargs: None

import warnings

import torch
import torch.distributed as dist  # 确保导入 dist
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import models.heter_csi as heter_csi
import models.heter_csi_moe as heter_csi_moe
import timm_utils.optim.optim_factory as optim_factory
import util.misc as misc
from engine_pretrain import train_one_epoch_3mask, train_one_epoch_csi
from util.data import *
from util.misc import NativeScalerWithGradNormCount as NativeScaler

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # ... [保留原有的参数] ...
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16_128', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='normal', type=str, help='model type used')
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
                        choices=['global', 'bucket', 'group'],
                        help='Whether to use fmow rgb, sentinel, or other dataset.')
    parser.add_argument('--bucket_num', default=4, type=int)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
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
    parser.add_argument('--data_num', default=1, type=float,
                        help='the amount of data used to finetune')

    # === [NEW] 新增参数用于仅计算Padding ===
    parser.add_argument('--calc_padding', action='store_true',
                        help='If true, only calculate padding ratio for 10 epochs and exit.')

    return parser

# === [NEW] 核心计算函数 ===
def measure_padding_efficiency(data_loader, device, num_epochs=10, logger=None):
    """
    计算前 num_epochs 的平均填充率。
    支持分布式计算，自动聚合所有Rank的结果。
    """
    if misc.is_main_process():
        print(f"\n[Efficiency Analysis] Starting padding ratio calculation for {num_epochs} epochs...")
        print(f"[Efficiency Analysis] Strategy: {data_loader.dataset.mmap_version} (Check shuffle type)")
    
    total_compute_tokens = 0.0  # 分母: BatchSize * MaxLen
    total_padding_tokens = 0.0  # 分子: (BatchSize * MaxLen) - Sum(ValidLengths)
    
    # 只需要遍历，不需要反向传播，不需要模型前向
    start_t = time.time()
    
    for epoch in range(num_epochs):
        # 设置 sampler epoch 以确保随机性一致
        if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
             data_loader.sampler.set_epoch(epoch)
        elif hasattr(data_loader, 'batch_sampler') and hasattr(data_loader.batch_sampler, 'set_epoch'):
             data_loader.batch_sampler.set_epoch(epoch)
             
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_data is None: continue
            
            # 解包 collate_fn 返回的数据: (padded_batch, lengths, dims)
            padded_inputs, lengths, _ = batch_data
            
            # --- 本地统计 ---
            B = padded_inputs.shape[0]
            max_len = padded_inputs.shape[1] # 当前batch的最大长度 h_j
            
            current_batch_total = B * max_len
            current_batch_valid = lengths.sum().item()
            current_batch_padding = current_batch_total - current_batch_valid
            
            total_compute_tokens += current_batch_total
            total_padding_tokens += current_batch_padding
    
    # --- 分布式聚合 (All Reduce) ---
    if dist.is_available() and dist.is_initialized():
        # 将本地统计量转换为 Tensor 放入 GPU
        stats_tensor = torch.tensor([total_padding_tokens, total_compute_tokens], device=device)
        # 求和所有 Rank 的结果
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        global_padding = stats_tensor[0].item()
        global_total = stats_tensor[1].item()
    else:
        global_padding = total_padding_tokens
        global_total = total_compute_tokens
        
    avg_padding_ratio = global_padding / global_total if global_total > 0 else 0.0
    elapsed = time.time() - start_t

    if misc.is_main_process():
        print("-" * 40)
        print(f"[Efficiency Analysis] Results over {num_epochs} epochs:")
        print(f"  > Total Tokens (Computed): {int(global_total):,}")
        print(f"  > Total Padding (Wasted):  {int(global_padding):,}")
        print(f"  > Padding Ratio:           {avg_padding_ratio:.4%} (Key Metric)")
        print(f"  > Time Elapsed:            {elapsed:.2f}s")
        print("-" * 40)
        
    return avg_padding_ratio


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if not args.distributed:
        args.gpu = "cuda:0"
    device = torch.device(args.gpu)
    # device = torch.device("cpu")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    mmap_name = f"{args.mask_type}_{args.shuffle_type}_{len(args.dataset)}_{args.bucket_num}_12345"
    # ... [Dataset 初始化代码保持不变] ...
    if args.distributed:
        if args.shuffle_type == 'global':
            dataset_train = CSIDataset_mmap_nopad(dataset=args.dataset, world_size=misc.get_world_size(),
                                            rank=misc.get_rank(), dataset_type='train', mmap_version=mmap_name, data_dir=args.data_dir)
        elif args.shuffle_type == 'bucket':
            dataset_train = CSIDataset_mmap_nopad(dataset=args.dataset, world_size=misc.get_world_size(),
                                            rank=misc.get_rank(), dataset_type='train', mmap_version=mmap_name, data_dir=args.data_dir)
        elif args.shuffle_type == 'group':
            dataset_train = CSIDataset_mmap_nopad(dataset=args.dataset, world_size=misc.get_world_size(),
                                            rank=misc.get_rank(), dataset_type='train', mmap_version=mmap_name, data_dir=args.data_dir)
    else:
        global_rank = 0
        dataset_train = CSIDataset_mmap_nopad(dataset=args.dataset, world_size=misc.get_world_size(),
                                        rank=misc.get_rank(), dataset_type='train', mmap_version=mmap_name, data_dir=args.data_dir)
    
    # ... [DataLoader 初始化代码保持不变] ...
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.shuffle_type == 'global':
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=False, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=CSIDataset_mmap_nopad.padded_collate_fn,
            # prefetch_factor=8,
        )
    elif args.shuffle_type == 'bucket':
        sampler_train = DistributedBucketBatchSampler_V2(
            dataset_bounds=dataset_train.dataset_bounds,
            batch_size=args.batch_size,
            accum_steps=args.accum_iter,
            num_buckets=args.bucket_num,
            world_size=num_tasks,
            rank=global_rank,
            shuffle=True,
            drop_last=True,
            seed=seed
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_sampler=sampler_train,
            collate_fn=CSIDataset_mmap_nopad.padded_collate_fn,
        )
    elif args.shuffle_type == 'group':
        sampler_train = DistributedGroupBatchSampler(
            dataset_bounds=dataset_train.dataset_bounds,
            batch_size=args.batch_size,
            world_size=num_tasks,
            rank=global_rank,
            shuffle=True,
            drop_last=True,
            seed=seed
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_sampler=sampler_train,
            collate_fn=CSIDataset_mmap_nopad.padded_collate_fn,
        )
    print("Sampler_train = %s" % str(sampler_train))
    
    # === [NEW] 在这里插入 Padding 计算逻辑 ===
    # 如果指定了 --calc_padding，则只计算并退出
    # 否则只打印一下，然后继续训练
    
    if args.calc_padding:
        # 只计算 Padding，不需要加载模型，节省显存和时间
        measure_padding_efficiency(data_loader_train, device, num_epochs=1)
        
        # 清理并退出
        if 'dataset_train' in locals():
            dataset_train.__del__() # 手动触发清理
        if dist.is_initialized():
            dist.destroy_process_group()
        return # 结束程序


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
