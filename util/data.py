import bisect
import pdb
import os
import math
import traceback

import h5py
import uuid
import shutil
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler, BatchSampler
import torch.distributed as dist
import numpy as np
import hdf5storage
from einops import rearrange
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy import random

import util.misc as misc


def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise


class CSIDataset_mmap_nopad(data.Dataset):
    def __init__(self,
                 dataset,
                 world_size=1,
                 rank=0,
                 dataset_type='train',
                 SNR=20,
                 patch_size=4,
                 data_num=None,
                 max_workers=2,
                 mmap_version='proposed',
                 data_dir='/mnt/4T/2/zcy/csidata'):
        super(CSIDataset_mmap_nopad, self).__init__()
        
        # 分布式信息
        self.world_size = world_size
        self.rank = rank
        
        # 基本参数
        self.patch_size = patch_size
        self.max_workers = max_workers
        self.dataset_type = dataset_type
        self.mmap_version = mmap_version
        self.data_dir = data_dir
        self.SNR = SNR
        
        # 处理数据集列表
        self.datasets_list = dataset.split(",")
        
        # 存储每个数据集的元数据
        self.dataset_bounds = []
        self.dataset_arrays = {}  # 存储每个数据集的mmap数组
        self.dataset_shapes = {}  # 存储每个数据集的形状
        
        # 1. 取消全局最大长度计算
        self._calculate_dataset_metadata(data_num)
        
        # 2. 创建内存映射文件
        self._create_dataset_mmap_files()
        
        # 3. 并行加载数据
        self._load_data_parallel()

        # 4. 全局数据集同步 (新增加)
        self._sync_datasets_across_ranks()
        
        # 等待所有进程完成加载
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        print(f"Rank {self.rank} passed data loading barrier")
            
    def _calculate_dataset_metadata(self, data_num):
        """计算每个数据集的元数据"""
        global_start = 0
        self.total_samples = 0
        
        for name in self.datasets_list:
            path = f"{self.data_dir}/{name}/{self.dataset_type}_data.mat"
            with h5py.File(path, 'r') as f:
                dset = f[f'H_{self.dataset_type}']
                U, K, T, B = dset.shape
                print(name)
                print("U, K, T, B:", U, K, T, B)
                if data_num is not None and data_num < B:
                    B = data_num
            
            # 序列长度计算（基于当前数据集）
            seq_length = T * K * U // (self.patch_size ** 3)
            
            self.dataset_bounds.append({
                'name': name,
                'path': path,
                'global_start': global_start,
                'global_end': global_start + B,
                'samples': B,
                'dims': (T, K, U),
                'seq_length': seq_length,
                'feature_dim': self.patch_size**3 * 2
            })
            
            self.total_samples += B
            global_start += B
        
        print(f"Total samples across all datasets: {self.total_samples}")
        
    def _create_dataset_mmap_files(self):
        """为每个数据集创建独立的内存映射文件"""
        if self.rank != 0:
            # 非主进程等待文件创建
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return
        
        # 主进程创建所有内存映射文件
        for meta in self.dataset_bounds:
            # 为每个数据集生成唯一文件路径
            mmap_path = os.path.join(
                self.data_dir, 
                f"csi_{self.dataset_type}_{meta['name']}_{self.mmap_version}.bin"
            )
            
            # 文件大小计算（精确匹配数据集实际形状）
            item_size = meta['seq_length'] * meta['feature_dim']
            file_size = meta['samples'] * item_size * np.dtype(np.float32).itemsize
            
            # 创建文件
            print(f"Creating memmap file for dataset {meta['name']}: "
                  f"{file_size/(1024**2):.2f} MB")
            with open(mmap_path, 'wb') as f:
                f.seek(file_size - 1)
                f.write(b'\0')
        
        # 文件创建完成，通知其他进程
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    
    def _load_data_parallel(self):
        # 计算每个rank负责的数据集范围
        datasets_per_rank = math.ceil(len(self.datasets_list) / self.world_size)
        start_idx = self.rank * datasets_per_rank
        end_idx = min((self.rank + 1) * datasets_per_rank, len(self.datasets_list))
        rank_datasets = [meta for meta in self.dataset_bounds 
                         if meta['name'] in self.datasets_list[start_idx:end_idx]]
    
        # print(f"---{self.rank}---")
        # print(rank_datasets)
        
        if not rank_datasets:
            print(f"Rank {self.rank} has no datasets to load")
            return
            
        print(f"Rank {self.rank} loading {len(rank_datasets)} datasets: "
              f"{[d['name'] for d in rank_datasets]}")
        
        # 创建线程池并行处理数据集
        with ThreadPoolExecutor(max_workers=min(len(rank_datasets), self.max_workers)) as executor:
            futures = [executor.submit(self._process_dataset, meta) for meta in rank_datasets]
            for fut in as_completed(futures):
                try:
                    fut.result()  # 如果线程里抛异常，这里会 re-raise
                except Exception as e:
                    print(f"[Rank {self.rank}] Error while processing dataset: {e}")
                    traceback.print_exc()

    def _sync_datasets_across_ranks(self):
        """确保所有rank都加载了全部数据集的内存映射"""
        # 等待所有rank完成数据处理
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        print(f"Rank {self.rank}: Syncing all datasets")
        
        # 每个rank加载全部数据集
        for meta in self.dataset_bounds:
            name = meta['name']
            mmap_path = os.path.join(
                self.data_dir, 
                f"csi_{self.dataset_type}_{name}_{self.mmap_version}.bin"
            )
            
            # 只读模式访问
            self.dataset_arrays[name] = np.memmap(
                mmap_path,
                dtype=np.float32,
                mode='r',
                shape=(meta['samples'], meta['seq_length'], meta['feature_dim'])
            )
            self.dataset_shapes[name] = meta['dims']
        
        print(f"Rank {self.rank}: Loaded {len(self.dataset_arrays)} datasets")
    
    def _process_dataset(self, meta):
        """加载并处理单个数据集"""
        name = meta['name']
        print(f"Rank {self.rank} processing dataset: {name}")
        
        # 数据集特定内存映射路径
        mmap_path = os.path.join(
            self.data_dir, 
            f"csi_{self.dataset_type}_{name}_{self.mmap_version}.bin"
        )
        
        # 打开内存映射文件 (r+ 模式)
        data_array = np.memmap(
            mmap_path,
            dtype=np.float32,
            mode='r+',
            shape=(meta['samples'], meta['seq_length'], meta['feature_dim'])
        )
        
        # 加载数据
        H_full = hdf5storage.loadmat(meta['path'])[f'H_{self.dataset_type}']
                
        # 功率归一化
        power = np.mean(np.abs(H_full)**2, axis=(1, 2, 3), keepdims=True)
        H_full = H_full / (np.sqrt(power))

        # 添加噪声
        if self.SNR is not None:
            noise = generate_gaussian_noise(H_full, self.SNR)
            H_full = H_full + noise

        # 转换为patch
        patched_data = patch_maker(H_full, self.patch_size)
        
        # 写入内存映射文件
        data_array[:, :, :] = patched_data
        data_array.flush()  # 确保数据写入磁盘
        
        # 关闭并重新以只读模式打开
        del data_array
        self.dataset_arrays[name] = np.memmap(
            mmap_path,
            dtype=np.float32,
            mode='r',
            shape=(meta['samples'], meta['seq_length'], meta['feature_dim'])
        )
        self.dataset_shapes[name] = meta['dims']
        
        print(f"Rank {self.rank} finished processing dataset: {name}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 找到对应数据集
        for meta in self.dataset_bounds:
            if meta['global_start'] <= idx < meta['global_end']:
                dataset_name = meta['name']
                local_idx = idx - meta['global_start']
                break
        else:
            raise IndexError(f"Index {idx} out of range")
        
        # 从内存映射获取数据
        data_arr = self.dataset_arrays[dataset_name][local_idx]
        actual_length = meta['seq_length']
        
        # 返回数据和元信息
        return torch.as_tensor(data_arr.copy()), actual_length, meta['dims']
    
    @staticmethod
    def padded_collate_fn(batch):
        """
        自动填充变长序列的批次处理函数
        参数:
            batch: [(data_tensor1, length1, feat_dim1), ...]
        """
        # 检查批次是否为空
        if not batch:
            return None
            
        # 解压数据、长度和特征维度
        data_list, lengths, feature_dims = zip(*batch)

        length = max(lengths)

        dims = torch.tensor(feature_dims)
        dims = dims.T
        
        # 创建填充后的批次张量
        padded_batch = torch.zeros(
            len(batch), 
            length,
            128,
            dtype=data_list[0].dtype
        )
        
        # 填充每个样本
        for i, (data, length, _) in enumerate(batch):
            padded_batch[i, :length] = data[:length]
            
        return padded_batch, torch.tensor(lengths), dims
    
    def __del__(self):
        """清理资源"""
        try:
            # 只有当所有rank都处理完毕时才删除文件
            if (dist.is_available() and dist.is_initialized() and self.rank == 0):
                dist.barrier()  # 确保所有rank已完成
                print(f"Rank {self.rank} deleting memmap files")
                for meta in self.dataset_bounds:
                    mmap_path = os.path.join(
                        self.data_dir, 
                        f"csi_{self.dataset_type}_{meta['name']}_{self.mmap_version}.bin"
                    )
                    if os.path.exists(mmap_path):
                        os.unlink(mmap_path)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")



def data_load_single_nopad(args, dataset_name, SNR=20, dataset_type='test', data_num=1): 

    folder_path = os.path.join(args.data_dir, f'{dataset_name}/{dataset_type}_data.mat')
    # folder_path = f'/data/zcy_data/zeroshot/{dataset_name}/{dataset_type}_data.mat'

    H_data = hdf5storage.loadmat(folder_path)[f'H_{dataset_type}']
    B, T, K, U = H_data.shape 
    
    if data_num < 1:
        target_len = int(B * data_num)
        H_data = H_data[:target_len]
        print(f"{dataset_name} sampled from {B} to {target_len} (ratio: {data_num})")
        B = target_len # 更新B的大小

    print(dataset_name, 'shape:', H_data.shape)

    power = np.mean(np.abs(H_data)**2, axis=(1, 2, 3), keepdims=True)
    H_data = H_data / (np.sqrt(power)) 
    
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)  

    H_data = patch_maker(H_data, 4)
    B, L, C = H_data.shape # 这里会获取切片后最新的B

    token_length = (T / 4) * (K / 4) * (U / 4)
    dataset_test = CSIDataset_Single(H_data, token_length, (T, K, U), dataset_name=dataset_name)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        # prefetch_factor=8,
    )
    return data_loader, L


class CSIDataset_Single(data.Dataset):
    def __init__(self, X_train, token_length, input_size, dataset_name):
        self.X_train = X_train
        self.token_length = token_length
        self.input_size = input_size
        self.dataset_name = dataset_name

    def __len__(self):

        return self.X_train.shape[0]

    def __getitem__(self, idx):

        return self.X_train[idx], self.token_length, self.input_size
    
    def get_dataset_name(self):
        return self.dataset_name


class WifoDataset(data.Dataset):
    def __init__(self, X_train, token_length, input_size, dataset_name):
        self.X_train = X_train
        self.token_length = token_length
        self.input_size = input_size
        self.dataset_name = dataset_name

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        return self.X_train[idx], self.token_length, self.input_size

    def get_dataset_name(self):
        return self.dataset_name


def data_load_single_Wifo(args, dataset, SNR=20): # 加载单个数据集

    folder_path_test = f'/data/zcy_data/CSIGPT_Dataset/test_data/{dataset}/test_data.mat'

    X_test = hdf5storage.loadmat(folder_path_test)['X_val']
    # X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex))
    H_data = X_test.transpose(0, 1, 3, 2)  # [B, T, U, K] -> [B, T, K, U]
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)   
    B, T, K, U = H_data.shape

    pdb.set_trace()
    H_data = patch_maker(H_data, 4).astype(np.float32)
    # 填充处理
    B, L, C = H_data.shape
    # max_length = args.max_length
    # # 检查长度一致性
    # if L > max_length:
    #     raise ValueError(f"Error in dataset: Sequence length {L} exceeds maximum length {max_length}.")
    
    padded_batch = np.zeros((B, L, C), dtype=H_data.dtype)
    padded_batch[:, :L, :] = H_data
    pdb.set_trace()
    test_data = WifoDataset(padded_batch, L, (T, K, U), dataset_name=dataset)

    batch_size = args.batch_size
    test_data = torch.utils.data.DataLoader(test_data, num_workers=args.num_workers, 
                                            batch_size = batch_size, shuffle=False, pin_memory=True, prefetch_factor=4)

    return test_data, L


def data_load(args, dataset_type, test_type='normal'):

    test_data_all = []
        
    for dataset_name in args.dataset.split(','):
        print(f"Processing {dataset_name} for {dataset_type}")
        if test_type == 'normal':
            # test_data, _ = data_load_single(args, dataset_name, dataset_type=dataset_type)
            test_data, _ = data_load_single_nopad(args, dataset_name, dataset_type=dataset_type, data_num=args.data_num)
        elif test_type == 'wifo':
            test_data, _ = data_load_single_Wifo(args, dataset_name)
        test_data_all.append(test_data)
    
    return test_data_all

def data_load_main(args, dataset_type='val', test_type='normal'):

    test_data = data_load(args, dataset_type, test_type)

    return test_data


def data_load_baseline(args, dataset_type='val', SNR=20, data_num=1.0):
    dataset_name = args.dataset
    folder_path = os.path.join(args.data_dir, f'{dataset_name}/{dataset_type}_data.mat')

    # 加载原始数据
    H_data = hdf5storage.loadmat(folder_path)[f'H_{dataset_type}']
    B, T, K, U = H_data.shape 
    print(f"{dataset_name} original shape: {H_data.shape}")

    # 如果 data_num 在 (0, 1) 之间，则进行采样
    if 0 < data_num < 1.0:
        # 计算需要保留的样本数量
        num_keep = int(B * data_num)
        
        if num_keep > 0:
            # 生成随机索引以打乱数据，避免只取前一部分可能导致的数据偏差
            # 如果不需要随机，可以直接用 H_data = H_data[:num_keep]
            perm_indices = np.random.permutation(B)
            selected_indices = perm_indices[:num_keep]
            H_data = H_data[selected_indices]
            
            # 更新 B 的大小，并打印提示信息
            B = H_data.shape[0]
            print(f"Finetuning sampling: kept {data_num*100:.1f}% data. New shape: {H_data.shape}")
        else:
            print("Warning: data_num implies 0 samples. Keeping original data.")

    power = np.mean(np.abs(H_data)**2, axis=(1, 2, 3), keepdims=True)
    H_data = H_data / (np.sqrt(power)) 
    
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)  

    dataset_test = CSIDataset_Single(H_data, 0, (T, K, U), dataset_name=dataset_name)
    
    # 注意：如果数据量减少了，RandomSampler 仍然会从现在的 dataset_test 中采样
    sampler = torch.utils.data.RandomSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True, # 如果采样后的数据量小于batch_size，这里可能会丢弃所有数据，需注意
        # prefetch_factor=8,
    )
    return data_loader

class DistributedBoundarySampler(Sampler):
    """
    适配多卡分布式训练的批次采样器
    
    特点:
    1. 保持子数据集边界完整 (不跨数据集分batch)
    2. 支持多卡分布式训练 (DDP模式)
    3. 全局批次洗牌 + 分片分配
    4. 同步最小batch数量确保训练稳定性
    """
    def __init__(self, 
                 dataset_bounds, 
                 batch_size, 
                 shuffle=True, 
                 seed=0,
                 rank=0,
                 world_size=1,
                 drop_last=True):
        """
        参数:
        dataset_bounds: List[dict] - 每个子数据集的边界信息
        batch_size: int - 每批次的样本数
        shuffle: bool - 是否洗牌
        seed: int - 随机种子
        rank: int - 当前进程ID (0 到 world_size-1)
        world_size: int - 总进程数 (GPU数量)
        drop_last: bool - 是否丢弃最后一个不完整的批次
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batch_indices = []
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        
        # 1. 按子数据集边界拆索引
        for bound in dataset_bounds:
            start, end = bound['global_start'], bound['global_end']
            indices = np.arange(start, end)
            
            # 按batch_size切分
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size].tolist()
                # 不丢弃或仅丢弃非完整batch
                if not drop_last or len(batch) == batch_size:
                    self.batch_indices.append(batch)
        
        # 2. 计算总批次数
        self.total_batches = len(self.batch_indices)
        
        # 3. 计算每个rank的批次分配
        self._calculate_rank_batches()
        
        # 4. 同步最小batch数量
        self.global_min_batches = self._sync_min_batches()
    
    def _calculate_rank_batches(self):
        """计算每个rank应分配的批次范围"""
        per_rank = self.total_batches // self.world_size
        extra = self.total_batches % self.world_size
        
        self.start_idx = self.rank * per_rank
        if self.rank < extra:
            self.start_idx += self.rank
            self.end_idx = self.start_idx + per_rank + 1
        else:
            self.start_idx += extra
            self.end_idx = self.start_idx + per_rank
        
        # 当前rank分配的批次数量
        self.rank_batch_count = self.end_idx - self.start_idx
    
    def _sync_min_batches(self):
        """同步所有rank的最小batch数量"""
        if self.world_size == 1:
            return self.rank_batch_count
            
        # 使用分布式操作计算最小batch数
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor(self.rank_batch_count, dtype=torch.int).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            return tensor.item()
        else:
            # 单机多卡模式，手动计算最小值
            min_count = self.rank_batch_count
            for r in range(self.world_size):
                if r == self.rank:
                    continue
                # 模拟其他rank的计算（实际应用中应使用分布式通信）
                per_rank = self.total_batches // self.world_size
                extra = self.total_batches % self.world_size
                rank_count = per_rank + (1 if r < extra else 0)
                min_count = min(min_count, rank_count)
            return min_count

    def set_epoch(self, epoch: int):
        """在每个epoch开始前由外部调用，保证多卡顺序一致"""
        self.epoch = epoch

    def __iter__(self):
        """生成当前rank应处理的批次"""
        # 创建洗牌副本以保持原始数据不变
        all_batches = self.batch_indices[:] 
        
        # 多卡一致洗牌
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(all_batches)
        
        # 当前rank分配的批次切片
        rank_batches = all_batches[self.start_idx:self.end_idx]
        
        # 只返回全局最小数量的批次
        for i in range(min(self.global_min_batches, len(rank_batches))):
            yield rank_batches[i]

    def __len__(self):
        """返回全局一致的batch数量"""
        return self.global_min_batches
    

class DistributedGroupBatchSampler(Sampler):
    def __init__(self, dataset_bounds, batch_size, world_size=None, rank=None, shuffle=True, drop_last=False, seed=42):
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_available() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() else 0

        self.group_bounds = dataset_bounds    
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0  # 添加epoch计数器
        
        # 从dataset获取组信息
        self._create_group_indices()
        self._create_global_index_map()
        
        # 分配样本到各rank（按长度升序排序）
        self._assign_samples_to_ranks()
        
        # 计算batch数量
        self.num_batches = self._calculate_num_batches()
        self.global_num_batches = self._sync_num_batches()
        
    def _create_group_indices(self):
        """创建组索引映射（合并相同长度的组）"""
        self.group_dict = {}
        for bound in self.group_bounds:
            length = bound['seq_length']
            indices = list(range(bound['global_start'], bound['global_end']))
            if length not in self.group_dict:
                self.group_dict[length] = []
            self.group_dict[length].extend(indices)
        
    def _create_global_index_map(self):
        """创建全局索引到序列长度的映射"""
        self.global_index_to_length = {}
        for length, indices in self.group_dict.items():
            for idx in indices:
                self.global_index_to_length[idx] = length
                
    def _assign_samples_to_ranks(self):
        """按长度升序分配样本"""
        # 合并所有样本并按长度升序排序
        all_samples = []
        for length, indices in sorted(self.group_dict.items(), key=lambda x: x[0]):
            if self.shuffle:
                # 使用与epoch无关的随机种子进行初始shuffle
                rng = np.random.RandomState(self.seed)
                rng.shuffle(indices)
            all_samples.extend(indices)
        
        total_samples = len(all_samples)
        
        # 计算每个rank分配的样本数量
        per_rank = total_samples // self.world_size
        remainder = total_samples % self.world_size
        
        # 分配样本
        start = 0
        self.rank_samples = []
        for i in range(self.world_size):
            end = start + per_rank + (1 if i < remainder else 0)
            self.rank_samples.append(all_samples[start:end])
            start = end
        
        # 打印分配信息
        if self.rank == 0:
            print("Sample distribution per rank:")
            for r in range(self.world_size):
                samples = self.rank_samples[r]
                min_len = self.global_index_to_length[samples[0]] if samples else 0
                max_len = self.global_index_to_length[samples[-1]] if samples else 0
                avg_len = (min_len + max_len) / 2 if samples else 0
                print(f"Rank {r}: {len(samples)} samples, "
                      f"min_len={min_len}, max_len={max_len}, avg_len={avg_len:.1f}")
    
    def _calculate_num_batches(self):
        """计算当前rank的batch数"""
        total_samples = len(self.rank_samples[self.rank])
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size
    
    def _sync_num_batches(self):
        """同步所有rank的batch数"""
        if self.world_size == 1:
            return self.num_batches
            
        # 使用分布式操作计算最小batch数
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor(self.num_batches, dtype=torch.int).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            return tensor.item()
        else:
            return self.num_batches
            
    def set_epoch(self, epoch):
        """设置当前epoch，用于随机种子生成"""
        self.epoch = epoch
    
    def __iter__(self):
        """生成索引列表批次"""
        # 获取当前rank的样本（已按长度升序排序）
        current_samples = self.rank_samples[self.rank]
        
        # 如果需要在批次级别打乱顺序，使用与epoch相关的随机种子
        if self.shuffle:
            # 使用epoch相关的随机种子确保每个epoch的shuffle不同
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(current_samples)
        
        # 创建批次
        batches = []
        for i in range(0, len(current_samples), self.batch_size):
            batch_indices = current_samples[i:i+self.batch_size]
            if not self.drop_last or len(batch_indices) == self.batch_size:
                batches.append(batch_indices)
        
        # 对batches进行shuffle（rank内batch级别的shuffle）
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch + 1)  # 使用不同的种子
            rng.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        return self.global_num_batches


class DistributedBucketBatchSampler(Sampler):
    def __init__(
        self,
        dataset_bounds,
        batch_size,
        accum_steps,
        num_buckets=2,
        world_size=None,
        rank=None,
        shuffle=True,
        drop_last=False,
        seed=0,
    ):
        if num_buckets is None or num_buckets < 1:
            raise ValueError("num_buckets must be a positive integer")

        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset_bounds = dataset_bounds
        self.seq_lengths = self._expand_seq_lengths_from_dataset_bounds(dataset_bounds)

        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

        self.bucket_boundaries = self._compute_bucket_boundaries(self.seq_lengths, self.num_buckets)
        self._create_buckets()                   # 保留空桶
        self._adjust_buckets_for_accumulation()  # 截断到 world_size*accum_steps 的倍数
        self._assign_samples_to_ranks()          # 按桶分配到各 rank

        self.num_batches = self._calculate_num_batches()
        self.global_num_batches = self._sync_num_batches()

    @staticmethod
    def _expand_seq_lengths_from_dataset_bounds(dataset_bounds):
        seq_lengths = []
        for entry in dataset_bounds:
            if 'samples' not in entry or 'seq_length' not in entry:
                raise ValueError("Each dataset_bounds entry must contain 'samples' and 'seq_length'.")
            s = int(entry['samples'])
            l = int(entry['seq_length'])
            seq_lengths.extend([l] * s)
        return seq_lengths

    @staticmethod
    def _compute_bucket_boundaries(seq_lengths, num_buckets):
        arr = np.asarray(seq_lengths)
        if num_buckets <= 1:
            return []
        perc = np.linspace(0, 100, num_buckets + 1)[1:-1]
        if len(perc) == 0:
            return []
        bounds = np.percentile(arr, perc)
        print("-------- Bucket Boundaries --------")
        print(bounds)
        return np.unique(bounds).tolist()

    def _create_buckets(self):
        num_bins = len(self.bucket_boundaries) + 1
        buckets = [[] for _ in range(num_bins)]
        for idx, length in enumerate(self.seq_lengths):
            bid = bisect.bisect_left(self.bucket_boundaries, length)
            bid = max(0, min(bid, num_bins - 1))
            buckets[bid].append(idx)
        self.buckets = buckets
        self.global_index_to_length = {i: self.seq_lengths[i] for i in range(len(self.seq_lengths))}

    def _adjust_buckets_for_accumulation(self):
        adjusted = []
        batches_per_cycle = self.world_size * self.accum_steps
        for bucket in self.buckets:
            if not bucket:
                adjusted.append([])
                continue
            num_batches = len(bucket) // self.batch_size
            keep_batches = (num_batches // batches_per_cycle) * batches_per_cycle
            adjusted.append(bucket[: keep_batches * self.batch_size] if keep_batches > 0 else [])
        self.buckets = adjusted
        if all(len(b) == 0 for b in self.buckets):
            raise ValueError("No buckets left after adjustment. Reduce num_buckets or accum_steps.")

    def _assign_samples_to_ranks(self):
        all_samples = []
        self.bucket_boundaries_indices = [0]
        for bid, bucket in enumerate(self.buckets):
            if self.shuffle and bucket:
                rng = np.random.RandomState(self.seed + self.epoch + bid)
                shuffled = bucket.copy()
                rng.shuffle(shuffled)
                all_samples.extend(shuffled)
            else:
                all_samples.extend(bucket.copy())
            self.bucket_boundaries_indices.append(len(all_samples))

        self.rank_samples = [[] for _ in range(self.world_size)]
        for b_idx in range(len(self.buckets)):
            start = self.bucket_boundaries_indices[b_idx]
            end = self.bucket_boundaries_indices[b_idx + 1]
            bucket_samples = all_samples[start:end]
            bucket_batches = len(bucket_samples) // self.batch_size
            if bucket_batches == 0:
                continue
            batches_per_rank = bucket_batches // self.world_size
            if batches_per_rank == 0:
                continue
            for r in range(self.world_size):
                s = start + r * batches_per_rank * self.batch_size
                e = s + batches_per_rank * self.batch_size
                self.rank_samples[r].extend(all_samples[s:e])

        if self.rank == 0:
            num_nonempty = sum(1 for b in self.buckets if len(b) > 0)
            print(f"\nBuckets: total={len(self.buckets)}, non-empty={num_nonempty}")
            print(f"Batch={self.batch_size}, accum_steps={self.accum_steps}, world_size={self.world_size}")
            for i, b in enumerate(self.buckets):
                if b:
                    avg = sum(self.seq_lengths[idx] for idx in b) / len(b)
                    print(f"  Bucket {i}: {len(b)} samples, avg_len={avg:.1f}")
                else:
                    print(f"  Bucket {i}: EMPTY")
            for r in range(self.world_size):
                s = self.rank_samples[r]
                if not s:
                    print(f"  Rank {r}: No samples")
                else:
                    lengths = [self.global_index_to_length[i] for i in s]
                    print(f"  Rank {r}: {len(s)} samples, batches={len(s)//self.batch_size}, min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    def _calculate_num_batches(self):
        total = len(self.rank_samples[self.rank])
        return total // self.batch_size if self.drop_last else math.ceil(total / self.batch_size)

    def _sync_num_batches(self):
        if self.world_size <= 1 or not dist.is_initialized():
            return self.num_batches
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = torch.tensor(self.num_batches, dtype=torch.int32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor.item())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.shuffle:
            self._assign_samples_to_ranks()
            self.num_batches = self._calculate_num_batches()
            self.global_num_batches = self._sync_num_batches()

    def __iter__(self):
        current = list(self.rank_samples[self.rank])
        if not current:
            return iter(())
        local_bucket_ids = [bisect.bisect_left(self.bucket_boundaries, self.global_index_to_length[i]) for i in current]

        boundaries = [(0, local_bucket_ids[0])]
        cur = local_bucket_ids[0]
        for i in range(1, len(local_bucket_ids)):
            if local_bucket_ids[i] != cur:
                cur = local_bucket_ids[i]
                boundaries.append((i, cur))
        boundaries.append((len(current), -1))

        bucket_batches = []
        for i in range(len(boundaries) - 1):
            s = boundaries[i][0]
            e = boundaries[i + 1][0]
            bid = boundaries[i][1]
            samples = current[s:e]
            batches_in_bucket = []
            for j in range(0, len(samples), self.batch_size):
                b = samples[j:j + self.batch_size]
                if len(b) == self.batch_size or not self.drop_last:
                    batches_in_bucket.append(b)
            if self.shuffle and batches_in_bucket:
                rng = np.random.RandomState(self.seed + self.epoch + bid)
                rng.shuffle(batches_in_bucket)
            bucket_batches.extend(batches_in_bucket)

        bucket_batches = bucket_batches[: self.global_num_batches]
        for batch in bucket_batches:
            yield batch

    def __len__(self):
        return int(self.global_num_batches)


# force one dataset per step for all ranks
class GroupBatchSampler(Sampler):
    """
    分组批次采样器 (支持多卡同步)

    特点:
    1. 每个 global step 上所有 rank 来自同一子数据集
    2. 支持全局 batch 级别 shuffle
    3. 保持子数据集边界，不跨数据集分 batch
    """
    def __init__(self, 
                 dataset_bounds, 
                 batch_size: int, 
                 shuffle: bool = True, 
                 seed: int = 0,
                 rank: int = 0,
                 world_size: int = 1,
                 drop_last: bool = False):

        assert world_size >= 1 and 0 <= rank < world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last

        dataset_batches = []
        for bound in dataset_bounds:
            start, end = bound['global_start'], bound['global_end']
            indices = np.arange(start, end)
            batches = []
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size].tolist()
                if not drop_last or len(batch) == batch_size:
                    batches.append(batch)
            dataset_batches.append(batches)

        global_rounds = []
        for batches in dataset_batches:
            if len(batches) == 0:
                continue
            n = len(batches)
            if n % self.world_size != 0:
                if self.drop_last:
                    batches = batches[: (n // self.world_size) * self.world_size]
                else:
                    need = self.world_size - (n % self.world_size)
                    pad = [batches[i % n] for i in range(need)]
                    batches = batches + pad
            for i in range(0, len(batches), self.world_size):
                chunk = batches[i:i + self.world_size]
                if len(chunk) == self.world_size:
                    global_rounds.append(chunk)

        self._global_rounds_base = global_rounds
        self.global_rounds_count = len(self._global_rounds_base)
        self.batches_per_rank = self.global_rounds_count

    def _get_epoch_rounds(self):
        rounds = list(self._global_rounds_base)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(rounds)
        return rounds

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rounds = self._get_epoch_rounds()
        for round_chunk in rounds:
            yield round_chunk[self.rank]

    def __len__(self):
        return self.batches_per_rank
    

def generate_gaussian_noise(data, snr_db):
    axes = tuple(range(1, data.ndim))
    signal_power = np.mean(np.abs(data) ** 2, axis=axes, keepdims=True)
    
    # Convert SNR to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Ensure SNR has proper shape for broadcasting
    if not isinstance(snr_linear, np.ndarray):
        snr_linear = np.array(snr_linear)
    if snr_linear.ndim == 0 or snr_linear.size == 1:
        snr_linear = snr_linear.reshape((-1,) + (1,)*(data.ndim-1))
    else:
        snr_linear = snr_linear.reshape((-1,) + (1,)*(data.ndim-1))
    
    # Calculate noise power
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    # Real and imaginary parts scaled appropriately
    noise_real = np.random.standard_normal(data.shape) * np.sqrt(noise_power / 2)
    noise_imag = np.random.standard_normal(data.shape) * np.sqrt(noise_power / 2)
    
    # Combine into complex noise
    noise = noise_real + 1j * noise_imag
    
    return noise


def patch_maker(data, patch_size=4):
    B, T, K, U = data.shape
    # 检查维度是否可被patch_size整除
    assert T % patch_size == 0 and K % patch_size == 0 and U % patch_size == 0, \
        "Dimensions must be divisible by patch_size"
    
    # 计算每个维度的块数
    t_blocks = T // patch_size
    k_blocks = K // patch_size
    u_blocks = U // patch_size
    
    # 将数据重组成块结构 [B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size]
    reshaped = data.reshape(B, 
                            t_blocks, patch_size, 
                            k_blocks, patch_size, 
                            u_blocks, patch_size)
    
    # 调整维度顺序：将块索引维度移到前面，内部patch维度移到后面
    # transposed = reshaped.permute(0, 1, 3, 5, 2, 4, 6)  # [B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size]
    transposed = reshaped.transpose(0, 1, 3, 5, 2, 4, 6) 
    
    # 合并块索引维度（所有块）和内部维度（每个patch展平）
    num_patches = t_blocks * k_blocks * u_blocks
    patch_elements = patch_size ** 3
    patched_data = transposed.reshape(B, num_patches, patch_elements)

    real_part = patched_data.real  
    imag_part = patched_data.imag  

    combined_patched_data = np.concatenate([real_part, imag_part], axis=-1) 
    
    return combined_patched_data


def patch_recover(patched_data, input_size, patch_size=4):
    """
    将 patch_maker 函数处理后的数据恢复为原始形状
    
    参数:
    patched_data: 经过 patch_maker 处理的数据，形状为 [B, num_patches, 2*(patch_size**3)]
    original_shape: 原始数据的形状 (B, T, K, U)
    patch_size: 块大小，与 patch_maker 中使用的一致
    
    返回:
    恢复后的复数数据，形状为 [B, T, K, U]
    """
    B = patched_data.shape[0]
    T, K, U = input_size
    T = T[0]
    K = K[0]
    U = U[0]

    # 计算每个维度的块数
    t_blocks = T // patch_size
    k_blocks = K // patch_size
    u_blocks = U // patch_size

    patched_data = patched_data[:, : t_blocks*k_blocks*u_blocks, :]
    
    # 拆分实部和虚部
    patch_elements = patch_size ** 3
    real_part = patched_data[..., :patch_elements]
    imag_part = patched_data[..., patch_elements:]
    
    # 重建复数数据
    complex_data = real_part + 1j * imag_part
        
    # 重塑为块结构 [B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size]
    reshaped = complex_data.reshape(B, t_blocks, k_blocks, u_blocks, 
                                   patch_size, patch_size, patch_size)
    
    # 调整维度顺序 - 逆操作
    transposed = reshaped.permute(0, 1, 4, 2, 5, 3, 6)
    
    # 合并块维度以恢复原始形状
    recovered = transposed.reshape(B, T, K, U)
    
    return recovered

def create_original_mask(mask_patch, input_size, patch_size):
    """
    将patch级别的mask转换为原始数据形状的mask
    
    参数:
    mask_patch: patch级别的mask，形状为 [B, num_patches]
    input_size: 原始数据的形状 (T, K, U)
    patch_size: patch大小
    
    返回:
    mask_original: 原始形状的mask，形状为 [B, T, K, U]
    """
    B = mask_patch.shape[0]
    T, K, U = input_size
    T = T[0]
    K = K[0]
    U = U[0]
    
    # 计算每个维度的块数
    t_blocks = T // patch_size
    k_blocks = K // patch_size
    u_blocks = U // patch_size
    
    # 将mask_patch reshape为块状结构 [B, t_blocks, k_blocks, u_blocks]
    mask_patch = mask_patch.view(B, t_blocks, k_blocks, u_blocks)
    
    # 扩展每个块的值到块内的每个元素
    mask_original = mask_patch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 添加三个维度
    mask_original = mask_original.expand(B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size)
    
    # 调整维度顺序以匹配原始数据布局（逆操作于patch_recover中的permute）
    mask_original = mask_original.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    mask_original = mask_original.view(B, T, K, U)
    
    return mask_original


if __name__ == "__main__":
    # Example usage
    # CSIDataset(file_path='/data/zcy_data/CSIGPT_Dataset/D1/train_data.mat', dataset_type='train')
    pass
