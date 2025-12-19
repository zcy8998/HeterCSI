import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# 引入 fvcore 进行 FLOPs 计算
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

# 引入你项目中的模块
import models.CSIGPT as CSIGPT
import util.misc as misc
from util.data import *
# 注意：不需要 engine_pretrain，因为我们只做前向推理

def get_args_parser():
    # 复用 main.py 的参数设置，确保环境一致
    parser = argparse.ArgumentParser('FLOPs Calculation', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='建议设为 1 以便精确观察每个样本的差异')
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='mae_vit_large_patch16_128', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=96, type=int)
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--mask_type', default='random', choices=['random', 'temporal', 'freq', 'all'])
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--spatial_mask', action='store_true', default=False)
    parser.add_argument('--cls_token', action='store_true', default=False)
    parser.add_argument('--depth', default=12, type=int)
    
    # Dataset parameters
    parser.add_argument('--shuffle_type', default='global', choices=['global', 'bucket', 'group'])
    parser.add_argument('--dataset', default="D1", type=str)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # Device
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    
    return parser

class ModelWrapper(torch.nn.Module):
    """
    包装器：fvcore 默认只传入 args，不传入 kwargs。
    我们需要将 mask_ratio 等参数固定在 forward 中，以便 fvcore 追踪图。
    """
    def __init__(self, model, mask_ratio, mask_type):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_type

    def forward(self, samples, token_length, input_size):
        # 这里的调用签名必须与 main.py 中一致
        return self.model(samples, token_length, input_size, 
                          mask_ratio=self.mask_ratio, 
                          mask_strategy=self.mask_strategy)

def main(args):
    # 1. 初始化环境 (简化的 misc.init_distributed_mode)
    # 如果是单卡测 FLOPs，不需要复杂的 DDP 初始化，强制单卡运行即可
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    print(f"--- 正在计算模式: {args.shuffle_type} ---")

    # 2. 构建 DataLoader (完全复用 main.py 逻辑)
    mmap_name = args.mask_type + args.shuffle_type
    
    print(args)
    # 这里假设只在单卡上跑 FLOPs 测试，world_size=1, rank=0
    dataset_train = CSIDataset_mmap_nopad(
        dataset=args.dataset, 
        world_size=1, 
        rank=0, 
        dataset_type='train', 
        mmap_version=mmap_name, 
        data_dir=args.data_dir
    )

    if args.shuffle_type == 'global':
        # Global 模式：通常会有 Padding 到最大长度
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=CSIDataset_mmap_nopad.padded_collate_fn,
        )
    elif args.shuffle_type == 'bucket':
        # Bucket 模式：使用 bucket sampler
        sampler_train = DistributedBucketBatchSampler(
            dataset_bounds=dataset_train.dataset_bounds,
            batch_size=args.batch_size,
            accum_steps=args.accum_iter,
            num_buckets=4,
            world_size=1, # 单卡计算
            rank=0,
            shuffle=False, # 测试时不需要 shuffle，顺序读即可
            drop_last=True,
            seed=args.seed
        )
        data_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_sampler=sampler_train,
            collate_fn=CSIDataset_mmap_nopad.padded_collate_fn,
        )
    else:
        raise NotImplementedError("此脚本暂只演示 global vs bucket")

    # 3. 构建模型
    model = CSIGPT.__dict__[args.model](
        cls_embed=args.cls_token,
        device=device
    )
    model.to(device)
    model.eval() # 设为 eval 模式

    # 包装模型以适配 fvcore
    model_wrapper = ModelWrapper(model, args.mask_ratio, args.mask_type)

    # 4. 计算 FLOPs
    total_flops = 0
    total_samples = 0
    
    print(f"开始遍历 DataLoader (Total Batches: {len(data_loader)})...")
    
    # 为了节省时间，如果你数据量很大，可以只跑前 N 个 batch
    # max_batches = 100 
    
    for idx, (samples, token_length, input_size) in enumerate(data_loader):
        samples = samples.to(device)
        token_length = token_length.to(device)
        # input_size 通常是 tensor 或 list，如果是 tensor 需要 to(device)
        if isinstance(input_size, torch.Tensor):
            input_size = input_size.to(device)

        # 构造 fvcore 输入元组
        inputs = (samples, token_length, input_size)

        # 核心：计算当前 Batch 的 FLOPs
        # fvcore 会追踪一次 forward 过程
        flops_counter = FlopCountAnalysis(model_wrapper, inputs)
        
        # 处理警告，某些操作符可能未被统计，通常是自定义操作，可忽略
        flops_counter.unsupported_ops_warnings(False) 

        batch_flops = flops_counter.total()
        batch_size = samples.shape[0]
        seq_len = samples.shape[1]

        total_flops += batch_flops
        total_samples += batch_size

        # 打印部分信息
        if idx % 10 == 0:
            print(f"Batch {idx}: Shape {samples.shape} | "
                  f"GFLOPs/img: {batch_flops / batch_size / 1e9:.4f}")

        # if idx >= max_batches: break

    # 5. 汇总结果
    avg_flops = total_flops / total_samples
    print("\n================ Results ================")
    print(f"Mode: {args.shuffle_type}")
    print(f"Mask Ratio: {args.mask_ratio}")
    print(f"Average GFLOPs per sample: {avg_flops / 1e9:.4f} G")
    print(f"Total parameters: {parameter_count_table(model_wrapper, max_depth=1)}")
    print("=========================================")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)