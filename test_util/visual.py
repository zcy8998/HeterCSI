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

import numpy as np

# pdb.set_trace = lambda *args, **kwargs: None

import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import models.CSIGPT as CSIGPT
import timm_utils.optim.optim_factory as optim_factory
import util.misc as misc
from engine_pretrain import train_one_epoch_3mask
from util.data import data_load_main
from util.metrics import NMSELoss
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.plot_util import CSIAttnVisualizer

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
    parser.add_argument('--resume_different_size', default='',
                        help='continue to pretrain for different size')
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
    parser.add_argument('--data_num', default=None, type=int,
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

    model = CSIGPT.__dict__[args.model](
        depth=args.depth,
        cls_embed=args.cls_token,
        device=device
    )
    
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank == 0:
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))
        total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    # 初始化可视化器
    visualizer = CSIAttnVisualizer(model, device=device)  # T=32, K=64, U=64

    for index, dataset_test in enumerate(dataset_val_all):
        dataset_name = dataset_test.dataset.get_dataset_name()
        for iteration, (samples, _, _) in enumerate(dataset_test, 1):
            # 可视化最后一个block的注意力
            samples = samples.to(device)
            visualizer.visualize_attention(
                samples,
                save_path=f"./{dataset_name}_sample{iteration}_attn.png",
            )
        break


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
