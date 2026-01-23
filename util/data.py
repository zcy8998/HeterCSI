import bisect
import pdb
import os
import math
import traceback

import h5py
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import numpy as np
import hdf5storage
from einops import rearrange
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
                 data_dir=None):
        super(CSIDataset_mmap_nopad, self).__init__()
        
        # Distributed information
        self.world_size = world_size
        self.rank = rank
        
        # Basic parameters
        self.patch_size = patch_size
        self.max_workers = max_workers
        self.dataset_type = dataset_type
        self.mmap_version = mmap_version
        self.data_dir = data_dir
        self.SNR = SNR
        
        # Process dataset list
        self.datasets_list = dataset.split(",")
        
        # Store metadata for each dataset
        self.dataset_bounds = []
        self.dataset_arrays = {}  # Store mmap array for each dataset
        self.dataset_shapes = {}  # Store shape for each dataset
        
        # 1. Calculate dataset metadata (replacing global max length calculation)
        self._calculate_dataset_metadata(data_num)
        
        # 2. Create memory-mapped files
        self._create_dataset_mmap_files()
        
        # 3. Load data in parallel
        self._load_data_parallel()

        # 4. Global dataset synchronization (newly added)
        self._sync_datasets_across_ranks()
        
        # Wait for all processes to finish loading
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        print(f"Rank {self.rank} passed data loading barrier")
            
    def _calculate_dataset_metadata(self, data_num):
        """Calculate metadata for each dataset"""
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
            
            # Sequence length calculation (based on current dataset)
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
        """Create independent memory-mapped files for each dataset"""
        if self.rank != 0:
            # Non-master processes wait for file creation
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return
        
        # Master process creates all memory-mapped files
        for meta in self.dataset_bounds:
            # Generate unique file path for each dataset
            mmap_path = os.path.join(
                self.data_dir, 
                f"csi_{self.dataset_type}_{meta['name']}_{self.mmap_version}.bin"
            )
            
            # File size calculation (exactly matching dataset actual shape)
            item_size = meta['seq_length'] * meta['feature_dim']
            file_size = meta['samples'] * item_size * np.dtype(np.float32).itemsize
            
            # Create file
            print(f"Creating memmap file for dataset {meta['name']}: "
                  f"{file_size/(1024**2):.2f} MB")
            with open(mmap_path, 'wb') as f:
                f.seek(file_size - 1)
                f.write(b'\0')
        
        # File creation complete, notify other processes
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    
    def _load_data_parallel(self):
        # Calculate the dataset range responsible for each rank
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
        
        # Create thread pool to process datasets in parallel
        with ThreadPoolExecutor(max_workers=min(len(rank_datasets), self.max_workers)) as executor:
            futures = [executor.submit(self._process_dataset, meta) for meta in rank_datasets]
            for fut in as_completed(futures):
                try:
                    fut.result()  # If an exception is thrown in the thread, it will re-raise here
                except Exception as e:
                    print(f"[Rank {self.rank}] Error while processing dataset: {e}")
                    traceback.print_exc()

    def _sync_datasets_across_ranks(self):
        """Ensure all ranks have loaded memory mappings for all datasets"""
        # Wait for all ranks to complete data processing
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        print(f"Rank {self.rank}: Syncing all datasets")
        
        # Each rank loads all datasets
        for meta in self.dataset_bounds:
            name = meta['name']
            mmap_path = os.path.join(
                self.data_dir, 
                f"csi_{self.dataset_type}_{name}_{self.mmap_version}.bin"
            )
            
            # Access in read-only mode
            self.dataset_arrays[name] = np.memmap(
                mmap_path,
                dtype=np.float32,
                mode='r',
                shape=(meta['samples'], meta['seq_length'], meta['feature_dim'])
            )
            self.dataset_shapes[name] = meta['dims']
        
        print(f"Rank {self.rank}: Loaded {len(self.dataset_arrays)} datasets")
    
    def _process_dataset(self, meta):
        """Load and process a single dataset"""
        name = meta['name']
        print(f"Rank {self.rank} processing dataset: {name}")
        
        # Dataset-specific memory mapping path
        mmap_path = os.path.join(
            self.data_dir, 
            f"csi_{self.dataset_type}_{name}_{self.mmap_version}.bin"
        )
        
        # Open memory-mapped file (r+ mode)
        data_array = np.memmap(
            mmap_path,
            dtype=np.float32,
            mode='r+',
            shape=(meta['samples'], meta['seq_length'], meta['feature_dim'])
        )
        
        # Load data
        H_full = hdf5storage.loadmat(meta['path'])[f'H_{self.dataset_type}']
                
        # Power normalization
        power = np.mean(np.abs(H_full)**2, axis=(1, 2, 3), keepdims=True)
        H_full = H_full / (np.sqrt(power))

        # Add noise
        if self.SNR is not None:
            noise = generate_gaussian_noise(H_full, self.SNR)
            H_full = H_full + noise

        # Convert to patches
        patched_data = patch_maker(H_full, self.patch_size)
        
        # Write to memory-mapped file
        data_array[:, :, :] = patched_data
        data_array.flush()  # Ensure data is written to disk
        
        # Close and reopen in read-only mode
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
        # Find corresponding dataset
        for meta in self.dataset_bounds:
            if meta['global_start'] <= idx < meta['global_end']:
                dataset_name = meta['name']
                local_idx = idx - meta['global_start']
                break
        else:
            raise IndexError(f"Index {idx} out of range")
        
        # Retrieve data from memory mapping
        data_arr = self.dataset_arrays[dataset_name][local_idx]
        actual_length = meta['seq_length']
        
        # Return data and meta info
        return torch.as_tensor(data_arr.copy()), actual_length, meta['dims']
    
    @staticmethod
    def padded_collate_fn(batch):
        """
        Batch processing function for padding variable-length sequences
        Args:
            batch: [(data_tensor1, length1, feat_dim1), ...]
        """
        # Check if batch is empty
        if not batch:
            return None
            
        # Unpack data, lengths, and feature dimensions
        data_list, lengths, feature_dims = zip(*batch)

        length = max(lengths)

        dims = torch.tensor(feature_dims)
        dims = dims.T
        
        # Create padded batch tensor
        padded_batch = torch.zeros(
            len(batch), 
            length,
            128,
            dtype=data_list[0].dtype
        )
        
        # Pad each sample
        for i, (data, length, _) in enumerate(batch):
            padded_batch[i, :length] = data[:length]
            
        return padded_batch, torch.tensor(lengths), dims
    
    def __del__(self):
        """Clean up resources"""
        try:
            # Delete files only when all ranks have finished processing
            if (dist.is_available() and dist.is_initialized() and self.rank == 0):
                dist.barrier()  # Ensure all ranks have completed
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
        B = target_len # Update size of B

    print(dataset_name, 'shape:', H_data.shape)

    power = np.mean(np.abs(H_data)**2, axis=(1, 2, 3), keepdims=True)
    H_data = H_data / (np.sqrt(power)) 
    
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)  

    H_data = patch_maker(H_data, 4)
    B, L, C = H_data.shape # Here we get the latest B after slicing/patching

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


def data_load_single_Wifo(args, dataset, SNR=20): # Load single dataset

    folder_path_test = f'/data/zcy_data/CSIGPT_Dataset/test_data/{dataset}/test_data.mat'

    X_test = hdf5storage.loadmat(folder_path_test)['X_val']
    # X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex))
    H_data = X_test.transpose(0, 1, 3, 2)  # [B, T, U, K] -> [B, T, K, U]
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)   
    B, T, K, U = H_data.shape

    pdb.set_trace()
    H_data = patch_maker(H_data, 4).astype(np.float32)
    # Padding processing
    B, L, C = H_data.shape
    
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

    # Load original data
    H_data = hdf5storage.loadmat(folder_path)[f'H_{dataset_type}']
    B, T, K, U = H_data.shape 
    print(f"{dataset_name} original shape: {H_data.shape}")

    # If data_num is between (0, 1), perform sampling
    if 0 < data_num < 1.0:
        # Calculate the number of samples to keep
        num_keep = int(B * data_num)
        
        if num_keep > 0:
            # Generate random indices to shuffle data, avoiding data bias caused by only taking the first part
            # If shuffling is not needed, you can directly use H_data = H_data[:num_keep]
            perm_indices = np.random.permutation(B)
            selected_indices = perm_indices[:num_keep]
            H_data = H_data[selected_indices]
            
            # Update the size of B and print prompt information
            B = H_data.shape[0]
            print(f"Finetuning sampling: kept {data_num*100:.1f}% data. New shape: {H_data.shape}")
        else:
            print("Warning: data_num implies 0 samples. Keeping original data.")

    power = np.mean(np.abs(H_data)**2, axis=(1, 2, 3), keepdims=True)
    H_data = H_data / (np.sqrt(power)) 
    
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)  

    dataset_test = CSIDataset_Single(H_data, 0, (T, K, U), dataset_name=dataset_name)
    
    sampler = torch.utils.data.RandomSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True, # If the data size after sampling is smaller than batch_size, all data might be dropped here, need attention
    )
    return data_loader


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
        self._create_buckets()                   
        self._adjust_buckets_for_accumulation()  
        self._assign_samples_to_ranks()          

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


class DistributedBucketBatchSampler_V2(Sampler):
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

        # Used to store debugging information
        self.global_index_to_length = {i: self.seq_lengths[i] for i in range(len(self.seq_lengths))}
        
        # 1. Create buckets (forced division into num_buckets)
        self._create_buckets()
        # 2. Adjust bucket size to adapt to gradient accumulation
        self._adjust_buckets_for_accumulation()
        # 3. Assign samples to each Rank and record interval information
        self._assign_samples_to_ranks()

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

    def _create_buckets(self):
        """
        Instead of using percentile to calculate boundaries, sort by length and split evenly.
        This ensures the number of buckets is strictly equal to num_buckets.
        """
        # Get all indices and sort by sequence length; stable sort prevents dataset confusion
        sorted_indices = np.argsort(self.seq_lengths, kind='stable')
        
        # Split the sorted indices into num_buckets parts
        # np.array_split can handle cases that are not divisible, distributing as evenly as possible
        self.buckets = [
            arr.tolist() 
            for arr in np.array_split(sorted_indices, self.num_buckets)
        ]
        
        # Print bucket information for verification
        if self.rank == 0:
            print(f"-------- Buckets Created (Total: {len(self.buckets)}) --------")

    def _adjust_buckets_for_accumulation(self):
        adjusted = []
        batches_per_cycle = self.world_size * self.accum_steps
        for bucket in self.buckets:
            if not bucket:
                adjusted.append([])
                continue
            num_batches = len(bucket) // self.batch_size
            # Ensure the number of batches in each bucket is a multiple of (world_size * accum_steps)
            keep_batches = (num_batches // batches_per_cycle) * batches_per_cycle
            adjusted.append(bucket[: keep_batches * self.batch_size] if keep_batches > 0 else [])
        self.buckets = adjusted
        
        # If all buckets are emptied, raise an error
        if all(len(b) == 0 for b in self.buckets):
            raise ValueError("No buckets left after adjustment. Reduce num_buckets or accum_steps.")

    def _assign_samples_to_ranks(self):
        """
        Assign samples in buckets to each Rank.
        Also record interval information of which original bucket the samples belong to in each Rank for use in __iter__.
        """
        # 1. Shuffle within each bucket (if enabled)
        all_samples_per_bucket = []
        for bid, bucket in enumerate(self.buckets):
            if self.shuffle and bucket:
                rng = np.random.RandomState(self.seed + self.epoch + bid)
                shuffled = bucket.copy()
                rng.shuffle(shuffled)
                all_samples_per_bucket.append(shuffled)
            else:
                all_samples_per_bucket.append(bucket.copy())

        # 2. Assign to each Rank
        self.rank_samples = [[] for _ in range(self.world_size)]
        
        # rank_bucket_intervals records which segment corresponds to which bucket in each rank's sample list.
        # Format: [(start_idx, end_idx, bucket_id), ...]
        self.rank_bucket_intervals = [[] for _ in range(self.world_size)]

        for b_idx, bucket_samples in enumerate(all_samples_per_bucket):
            bucket_len = len(bucket_samples)
            bucket_batches = bucket_len // self.batch_size
            
            if bucket_batches == 0:
                continue
                
            batches_per_rank = bucket_batches // self.world_size
            if batches_per_rank == 0:
                continue
                
            for r in range(self.world_size):
                s = r * batches_per_rank * self.batch_size
                e = s + batches_per_rank * self.batch_size
                
                # Get the sample fragment assigned to the current rank
                rank_slice = bucket_samples[s:e]
                
                # Record the start position in rank_samples
                start_pos = len(self.rank_samples[r])
                self.rank_samples[r].extend(rank_slice)
                end_pos = len(self.rank_samples[r])
                
                # Record interval information: (start index, end index, original bucket ID)
                self.rank_bucket_intervals[r].append((start_pos, end_pos, b_idx))

        # 3. Print statistical information
        if self.rank == 0:
            num_nonempty = sum(1 for b in self.buckets if len(b) > 0)
            print(f"\nBuckets: total={len(self.buckets)}, non-empty={num_nonempty}")
            print(f"Batch={self.batch_size}, accum_steps={self.accum_steps}, world_size={self.world_size}")
            
            # Briefly print bucket information
            for i, b in enumerate(self.buckets):
                if b:
                    avg = sum(self.seq_lengths[idx] for idx in b) / len(b)
                    # Can also print min/max to confirm correct sorting
                    lens = [self.seq_lengths[idx] for idx in b]
                    print(f"  Bucket {i}: {len(b)} samples, len_range=[{min(lens)}, {max(lens)}], avg={avg:.1f}")
                else:
                    print(f"  Bucket {i}: EMPTY (Filtered by accumulation adjustment)")

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
        """
        New iteration logic: directly use the bucket intervals calculated in _assign_samples_to_ranks.
        No longer rely on bisect lookup, avoiding the issue of fuzzy boundaries.
        """
        current_rank_samples = self.rank_samples[self.rank]
        intervals = self.rank_bucket_intervals[self.rank]
        
        if not current_rank_samples:
            return iter(())

        bucket_batches = []
        
        # Iterate through the intervals of each bucket
        for (start, end, bid) in intervals:
            samples = current_rank_samples[start:end]
            
            # Slice the samples in this interval into batches
            batches_in_bucket = []
            for j in range(0, len(samples), self.batch_size):
                b = samples[j : j + self.batch_size]
                if len(b) == self.batch_size or not self.drop_last:
                    batches_in_bucket.append(b)
            
            # If Shuffle is needed, shuffle the order of these batches
            # (Note: samples were already shuffled in _assign_samples_to_ranks, here we shuffle the output order of batches)
            if self.shuffle and batches_in_bucket:
                rng = np.random.RandomState(self.seed + self.epoch + bid)
                rng.shuffle(batches_in_bucket)
            
            bucket_batches.extend(batches_in_bucket)

        # Truncate to the global minimum number of batches to prevent multi-card training deadlocks
        bucket_batches = bucket_batches[: self.global_num_batches]
        
        for batch in bucket_batches:
            yield batch

    def __len__(self):
        return int(self.global_num_batches)
    

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
        self.epoch = 0  # Add epoch counter
        
        # Get group information from dataset
        self._create_group_indices()
        self._create_global_index_map()
        
        # Assign samples to each rank (sorted by length in ascending order)
        self._assign_samples_to_ranks()
        
        # Calculate number of batches
        self.num_batches = self._calculate_num_batches()
        self.global_num_batches = self._sync_num_batches()
        
    def _create_group_indices(self):
        """Create group index mapping (merge groups with the same length)"""
        self.group_dict = {}
        for bound in self.group_bounds:
            length = bound['seq_length']
            indices = list(range(bound['global_start'], bound['global_end']))
            if length not in self.group_dict:
                self.group_dict[length] = []
            self.group_dict[length].extend(indices)
        
    def _create_global_index_map(self):
        """Create mapping from global index to sequence length"""
        self.global_index_to_length = {}
        for length, indices in self.group_dict.items():
            for idx in indices:
                self.global_index_to_length[idx] = length
                
    def _assign_samples_to_ranks(self):
        """Assign samples in ascending order of length"""
        # Merge all samples and sort by length in ascending order
        all_samples = []
        for length, indices in sorted(self.group_dict.items(), key=lambda x: x[0]):
            if self.shuffle:
                # Use random seed independent of epoch for initial shuffle
                rng = np.random.RandomState(self.seed)
                rng.shuffle(indices)
            all_samples.extend(indices)
        
        total_samples = len(all_samples)
        
        # Calculate the number of samples assigned to each rank
        per_rank = total_samples // self.world_size
        remainder = total_samples % self.world_size
        
        # Assign samples
        start = 0
        self.rank_samples = []
        for i in range(self.world_size):
            end = start + per_rank + (1 if i < remainder else 0)
            self.rank_samples.append(all_samples[start:end])
            start = end
        
        # Print distribution information
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
        """Calculate number of batches for current rank"""
        total_samples = len(self.rank_samples[self.rank])
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size
    
    def _sync_num_batches(self):
        """Sync number of batches across all ranks"""
        if self.world_size == 1:
            return self.num_batches
            
        # Use distributed operations to calculate the minimum number of batches
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor(self.num_batches, dtype=torch.int).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            return tensor.item()
        else:
            return self.num_batches
            
    def set_epoch(self, epoch):
        """Set current epoch, used for random seed generation"""
        self.epoch = epoch
    
    def __iter__(self):
        """Generate batches of index lists"""
        # Get samples for current rank (already sorted by length in ascending order)
        current_samples = self.rank_samples[self.rank]
        
        # If shuffling at batch level is needed, use random seed related to epoch
        if self.shuffle:
            # Use epoch-related random seed to ensure different shuffle every epoch
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(current_samples)
        
        # Create batches
        batches = []
        for i in range(0, len(current_samples), self.batch_size):
            batch_indices = current_samples[i:i+self.batch_size]
            if not self.drop_last or len(batch_indices) == self.batch_size:
                batches.append(batch_indices)
        
        # Shuffle batches (shuffle at batch level within rank)
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch + 1)  # Use a different seed
            rng.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        return self.global_num_batches
    
        
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
    # Check if dimensions are divisible by patch_size
    assert T % patch_size == 0 and K % patch_size == 0 and U % patch_size == 0, \
        "Dimensions must be divisible by patch_size"
    
    # Calculate number of blocks for each dimension
    t_blocks = T // patch_size
    k_blocks = K // patch_size
    u_blocks = U // patch_size
    
    # Reshape data into block structure [B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size]
    reshaped = data.reshape(B, 
                            t_blocks, patch_size, 
                            k_blocks, patch_size, 
                            u_blocks, patch_size)
    
    # Adjust dimension order: move block indices to the front, internal patch dimensions to the back
    # transposed = reshaped.permute(0, 1, 3, 5, 2, 4, 6)  # [B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size]
    transposed = reshaped.transpose(0, 1, 3, 5, 2, 4, 6) 
    
    # Merge block index dimensions (all blocks) and internal dimensions (flatten each patch)
    num_patches = t_blocks * k_blocks * u_blocks
    patch_elements = patch_size ** 3
    patched_data = transposed.reshape(B, num_patches, patch_elements)

    real_part = patched_data.real  
    imag_part = patched_data.imag  

    combined_patched_data = np.concatenate([real_part, imag_part], axis=-1) 
    
    return combined_patched_data


if __name__ == "__main__":
    pass
