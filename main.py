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

    return parser


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
    # max_length = dataset_train.max_length

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
    # elif args.shuffle_type == 'bucket':
    #     sampler_train = DistributedBoundarySampler(
    #         dataset_bounds=dataset_train.dataset_bounds,
    #         batch_size=args.batch_size,
    #         world_size=num_tasks,
    #         rank=global_rank,
    #         shuffle=True,
    #         drop_last=True,
    #         seed=seed
    #     )
    #     data_loader_train = torch.utils.data.DataLoader(
    #         dataset_train,
    #         batch_sampler=sampler_train,
    #     )
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
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0:
        dataset_val_all = data_load_main(args, dataset_type='val', test_type='normal') # 加载数据

    model = None
    if args.model_type == 'normal':
        model = heter_csi.__dict__[args.model](
            cls_embed=args.cls_token,
            device=device
        )
    elif args.model_type == 'moe':    
        model = heter_csi_moe.__dict__[args.model](
        cls_embed=args.cls_token,
        device=device
    )   
 
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume:
        print("Start finetune, resume from checkpoint: %s" % args.resume)
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # misc.load_model_different_size(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank == 0:
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))
        total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    total_training_time = 0.0 

    if misc.is_main_process() and args.output_dir:
        results_file = os.path.join(args.output_dir, f"nmse_results_{args.mask_type}_{args.shuffle_type}.csv")
        time_results_file = os.path.join(args.output_dir, f"nmse_results_pure_time_{args.mask_type}_{args.shuffle_type}.csv") # 改个名区分一下
        if args.resume:
            results_file += "_finuetune"
        with open(results_file, "w") as f:
            f.write("epoch,mask_type,avg_nmse\n")
        with open(time_results_file, "w") as f:
            f.write("train_time,mask_type,avg_nmse\n")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            if args.shuffle_type == 'global':
                data_loader_train.sampler.set_epoch(epoch)
            elif args.shuffle_type == 'bucket' or args.shuffle_type == 'group':
                data_loader_train.batch_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        if args.mask_type == 'all':
            train_one_epoch_3mask(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
        else:
            train_one_epoch_csi(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time += epoch_duration
        epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_end_time - epoch_start_time)))
        print(f"Epoch {epoch}, time consume {epoch_total_time_str}")

        if args.output_dir and ((epoch+1) % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if args.output_dir and misc.is_main_process() and ((epoch+1) % 10 == 0 or epoch + 1 == args.epochs):
            if args.mask_type == 'all':
                mask_list = {'random': 0.85, 'temporal': 0.5, 'freq': 0.5}
            else:
                mask_list = {args.mask_type: args.mask_ratio}
            epoch_results = []
            model.eval()
            for task_idx, (mask_type, mask_ratio) in enumerate(mask_list.items()):
                nmse_list = []
                # 初始化任务统计量
                for index, dataset_test in enumerate(dataset_val_all):
                    dataset_name = dataset_test.dataset.get_dataset_name()
                    print(f"Start test {dataset_name}.")
                    with torch.no_grad():
                        error_nmse = 0
                        num = 0
                        epoch_val_loss = []
                        for _, (samples, token_length, input_size) in enumerate(dataset_test, 1):
                            optimizer.zero_grad()  # fixed
                            samples = samples.to(device)
                            token_length = token_length.to(device)
                            pdb.set_trace()
                            loss, pred, mask = model(samples, token_length, input_size, 
                                                    mask_ratio=mask_ratio, mask_strategy=mask_type)

                            N, L, D = samples.shape
                            col_indices = torch.arange(L, device=samples.device).expand(N, L)
                            mask_in_length = col_indices < token_length[:, None]
                            bool_mask = mask.bool()
                            mask_nmse = bool_mask & mask_in_length

                            N = pred.shape[0]
                            y_pred = pred[mask_nmse==1].reshape(-1,1).reshape(N,-1).detach().cpu().numpy()  # [Batch_size, 样本点数目]
                            y_target = samples[mask_nmse==1].reshape(-1,1).reshape(N,-1).detach().cpu().numpy()

                            error_nmse += np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
                            num += y_pred.shape[0]
                            epoch_val_loss.append(loss.item())  # save all losses into a vector for one epoch

                        nmse = error_nmse / num
                        v_loss = np.nanmean(np.array(epoch_val_loss))
                        nmse_list.append(nmse)
                        print(f'dataset_name: {dataset_name}, Validation loss: {v_loss:.7f}, NMSE: {nmse:.7f}')   

                avg_nmse = np.mean(nmse_list)
                epoch_results.append((mask_type, avg_nmse))

                print("------------------------------")
                print(f"Task type is {mask_type}, Average NMSE: {avg_nmse:.7f}")

                with open(results_file, "a") as f:
                    f.write(f"{epoch},{mask_type},{avg_nmse:.7f}\n")
                
                with open(time_results_file, "a") as f:
                    # 这里的 total_training_time_min 是截止到当前 Epoch 训练结束时的纯训练累计时间
                    f.write(f"{total_training_time:.4f},{mask_type},{avg_nmse:.7f}\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if misc.is_main_process():
        # 计算实际运行的 epoch 数量
        num_trained_epochs = args.epochs - args.start_epoch
        
        if num_trained_epochs > 0:
            # 计算平均每个 epoch 的秒数
            avg_time_per_epoch_sec = total_training_time / num_trained_epochs
            # 转换为分钟
            avg_time_per_epoch_min = avg_time_per_epoch_sec / 60
            
            print("-" * 30)
            print(f"Training finished.")
            print(f"Total training time: {total_training_time/60:.2f} minutes")
            print(f"Average training time per epoch: {avg_time_per_epoch_min:.4f} minutes")
            print("-" * 30)

    if 'train_dataset' in locals():
        dataset_train.cleanup()
    
    # 销毁分布式组
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
