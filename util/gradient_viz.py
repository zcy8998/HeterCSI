import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


class GradientSaver:
    """
    Save the flattened, L2-normalized gradient vector (numpy float32) to disk.
    Files are written to: save_root/<dataset_name>/grad_epoch{epoch}_step{step}.npy
    """

    def __init__(self, save_root: str, mkdirs: bool = True):
        self.save_root = Path(save_root)
        if mkdirs:
            self.save_root.mkdir(parents=True, exist_ok=True)

    def save_from_model(self, model: torch.nn.Module, dataset_name: str, epoch: int, step: int, is_norm=True, task="random", batch_size=16) -> str:
        """
        Read gradients from model.parameters().grad, flatten to a single vector, normalize (L2),
        and save as .npy. Returns path to saved file or empty string if nothing saved.
        NOTE: call this AFTER loss.backward() and BEFORE optimizer.step() (so .grad exists).
        """
        device_cpu = torch.device("cpu")
        grads_list = []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = p.grad
            if g is None:
                continue
            grads_list.append(g.detach().to(device_cpu).view(-1))

        if len(grads_list) == 0:
            return ""

        flat = torch.cat(grads_list)  # 1-D CPU tensor
        arr = flat.numpy().astype(np.float32)

        if is_norm:
            norm = np.linalg.norm(arr)
            if norm > 0.0:
                arr = arr / norm  # L2-normalize
                
        # save file
        out_dir = self.save_root / dataset_name / f"{task}_{batch_size}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"grad_epoch{epoch:04d}_step{step:06d}.npy"
        np.save(str(fname), arr)
        return str(fname)
    



    def _collect_grads_and_magnitude(self, model: torch.nn.Module):
        """收集梯度，计算幅值，并进行归一化"""
        grads_list = []
        for p in model.parameters():
            if p.grad is not None:
                grads_list.append(p.grad.detach().cpu().reshape(-1))
            elif p.requires_grad:
                grads_list.append(torch.zeros_like(p).detach().cpu().reshape(-1))
        
        if not grads_list:
            return None, None

        # 拼接成一个大向量
        flat = torch.cat(grads_list)
        arr = flat.numpy().astype(np.float32)
        
        # 计算幅值 (Magnitude)
        magnitude = np.linalg.norm(arr)
        
        # 归一化 (Direction)
        if magnitude > 1e-8:
            arr_norm = arr / magnitude
        else:
            arr_norm = arr # 0向量保持不变

        return arr_norm, magnitude

    def save_per_sample_batch_grads(self,
                                    model: torch.nn.Module,
                                    batch,
                                    dataset_name: str,
                                    epoch: int,
                                    step: int,
                                    device=None,
                                    args=None):
        # --- 1. 解析 Batch ---
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
            token_lengths = batch[1] if len(batch) > 1 else None
            input_size = batch[2]
        else:
            imgs = batch
            token_lengths = None

        # --- 2. 准备设备 ---
        if device is None:
            device = next(model.parameters()).device
        
        imgs = imgs.to(device, non_blocking=True)
        if token_lengths is not None:
            token_lengths = token_lengths.to(device, non_blocking=True)
        
        model.eval()
        model.zero_grad()

        out_dir = self.save_root / args.shuffle_type
        out_dir.mkdir(parents=True, exist_ok=True)

        batch_size = imgs.shape[0]
        saved_count = 0
        input_size = input_size.permute(1,0)

        # --- 3. 逐样本循环 ---
        for i in range(batch_size):
            img_i = imgs[i].unsqueeze(0)
            len_i = token_lengths[i].unsqueeze(0) if token_lengths is not None else None

            # Forward (传入 token_length)
            if len_i is not None:
                output = model(img_i, token_length=len_i, input_size=input_size[i].unsqueeze(1), mask_ratio=args.mask_ratio, mask_strategy=args.mask_type)
            else:
                output = model(img_i)

            # Extract Loss
            loss = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()

            if loss is None:
                continue

            # Backward
            loss.backward()

            # --- 核心修改：获取方向和幅值 ---
            grad_norm, grad_mag = self._collect_grads_and_magnitude(model)
            
            if grad_norm is not None:
                # 文件名格式化
                base_name = f"epoch{epoch:04d}_step{step:06d}_sample{i:04d}"
                
                # 保存归一化后的梯度向量 (方向)
                np.save(str(out_dir / f"grad_{base_name}.npy"), grad_norm)
                
                # 保存梯度幅值 (标量)
                np.save(str(out_dir / f"mag_{base_name}.npy"), np.array(grad_mag))
                
                saved_count += 1

            model.zero_grad()

        return saved_count, str(out_dir)