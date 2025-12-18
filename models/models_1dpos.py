# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import pdb
import math

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from util import video_vit
from util.comm_util import delay_doppler_phy_loss
from util.data import patch_recover
from util.logging import master_print as print
from util.pos_embed import get_1d_sincos_pos_embed_from_grid


class MaskedAutoencoderCSI(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
            self,
            max_length=1024,
            embed_dim=1024,
            depth=8,
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=4,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            norm_pix_loss=False,
            patch_embed=video_vit.PatchEmbed_v2,
            no_qkv_bias=False,
            trunc_init=False,
            cls_embed=False,
            pred_t_dim=9,
            device=None,
            **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim

        self.max_length = max_length
        self.patch_embed = patch_embed()
        input_size = (4, 4, 4, 2)
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if self.cls_embed:
            _num_patches = self.max_length + 1
        else:
            _num_patches = self.max_length
        pos_embed_np = get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(_num_patches))
        self.pos_embed = torch.tensor(pos_embed_np, dtype=torch.float32).unsqueeze(0).to(device)

        self.blocks = nn.ModuleList(
            [
                video_vit.Block_v2(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        decoder_pos_embed_np = get_1d_sincos_pos_embed_from_grid(decoder_embed_dim, np.arange(_num_patches))
        self.decoder_pos_embed = torch.tensor(decoder_pos_embed_np, dtype=torch.float32).unsqueeze(0).to(device)

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block_v2(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.input_size[0] * self.input_size[1] * self.input_size[2] * self.input_size[3],
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, C, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert W % p == 0 and H % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u

        x = imgs.reshape(shape=(N, C, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p ** 2 * C))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # pdb.set_trace()
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep
    
    def temporal_masking(self, x, input_size, mask_ratio=0.5):
        """
        向量化的时序掩码处理，模拟随机掩码效果但实际保留连续时间段
        Parameters:
            x: 输入数据，(B, max_length, D)
            input_size: 每个样本的[时间, 频率, 天线]维度，(B, 3)
            mask_ratio: 掩码比例，默认0.5
        Returns:
            x_masked: 掩码后数据，(B, L_keep_max, D)
            mask: 掩码矩阵，(B, max_length)
            ids_restore: 原始索引，(B, max_length)
            ids_keep: 保留位置的索引，(B, L_keep_max)
        """
        B, max_length, D = x.shape
        device = x.device
        t, k, u = input_size
        # t, k, u = torch.unbind(input_size, dim=1)
        t = t.to(device)
        k = k.to(device)
        u = u.to(device)
        tb, kb, ub = t // 4, k // 4, u // 4         # 分块处理(假设4x4块)
        patches_per_t = kb * ub                     # 每时间块的块数 (B,)
        t_keep = (tb * (1 - mask_ratio)).long()     # 需保留的时间块数 (B,)
        L_keep_sample = t_keep * patches_per_t      # 各样本实际保留块数 (B,)
        pdb.set_trace()
        # 2) 构建时序索引矩阵
        arange = torch.arange(max_length, device=device)  # (0到max_length-1)
        # 计算每个位置属于的时间块索引 (B, max_length)
        time_idx = arange.unsqueeze(0) // patches_per_t.unsqueeze(1)

        # 3) 生成保留掩码
        keep_mask = time_idx < t_keep.unsqueeze(1)  # 前t_keep个时间块保留 (B, max_length)
        mask = (~keep_mask).to(torch.float32)       # 转换为浮点掩码 (B, max_length)

        # 4) 准备索引恢复矩阵
        ids_restore = arange.unsqueeze(0).expand(B, -1)  # (B, max_length)

        # 5) 生成保留索引（考虑不同样本保留块数不同）
        filler = torch.full_like(ids_restore, max_length)  # 超限填充值 (B, max_length)
        ids_filled = torch.where(keep_mask, ids_restore, filler)  # 保留位置填原索引，其余填max_length
        
        # 计算最大保留块数（补齐不同样本长度差异）
        L_keep_max = L_keep_sample.max().long()
        # 排序后将保留索引集中在前部 (B, max_length)
        ids_sorted, _ = torch.sort(ids_filled, dim=1)
        ids_keep = ids_sorted[:, :L_keep_max]  # 截取最大保留块数 (B, L_keep_max)

        # 6) 安全索引处理（避免超限索引导致错误）
        is_valid = ids_keep < max_length  # 标记有效索引 (B, L_keep_max)
        ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep))  # 超限位置置0

        # 7) 索引提取数据
        idx = ids_safe.unsqueeze(-1).expand(-1, -1, D)  # 扩展至特征维度 (B, L_keep_max, D)
        x_masked = torch.gather(x, dim=1, index=idx)    # 收集数据 (B, L_keep_max, D)
        
        # 8) 将填充位置清零
        zero_mask = is_valid.unsqueeze(-1).expand(-1, -1, D)  # 有效性掩码 (B, L_keep_max, D)
        x_masked = x_masked * zero_mask.to(x_masked.dtype)    # 无效位置清零

        return x_masked, mask, ids_restore, ids_keep, is_valid
    
    def freq_masking(self, x, input_size, mask_ratio=0.5):
        """
        向量化的频率掩码处理，修正版 - 正确保留低频部分
        """
        B, max_length, D = x.shape
        device = x.device
        
        t, k, u = input_size
        # t, k, u = torch.unbind(input_size, dim=1)
        t = t.to(device)
        k = k.to(device)
        u = u.to(device)
        # 计算块数量 (向量化处理)
        tb = (t // 4).long()  # 时间块数
        kb = (k // 4).long()  # 频率块数
        ub = (u // 4).long()  # 空间块数
        total_blocks = tb * kb * ub
        
        # ===== 核心修正：向量化生成频率块索引 =====
        # 创建全局索引 (0 到 max_length-1)
        global_idx = torch.arange(max_length, device=device).unsqueeze(0)  # (1, L)
        
        # 计算每个位置的块坐标 (向量化坐标分解)
        kf_tmp = global_idx // ub.unsqueeze(1)           # 中间值
        k_idx = kf_tmp % kb.unsqueeze(1)                 # 频率块索引 (B, L)
        
        # 标记有效位置 (实际存在的块)
        valid_mask = global_idx < total_blocks.unsqueeze(1)  # (B, L)
        
        # 计算需保留的频率块数 (确保至少保留1个块)
        k_keep = (kb * (1 - mask_ratio)).round().long()    # (B,)
        k_keep = torch.maximum(k_keep, torch.ones_like(k_keep))
        
        # 构建保留掩码 (只保留低频块且位置有效)
        keep_mask = (k_idx < k_keep.unsqueeze(1)) & valid_mask  # (B, L)
        mask = (~keep_mask).to(torch.float32)  # 转换为浮点掩码
        
        # 准备索引恢复矩阵
        ids_restore = torch.arange(max_length, device=device).unsqueeze(0).expand(B, -1)
        
        # 生成保留索引（考虑不同样本保留块数不同）
        filler = torch.full_like(ids_restore, max_length)  # 超限填充值
        ids_filled = torch.where(keep_mask, ids_restore, filler)
        
        # 计算最大保留块数（补齐样本长度差异）
        L_keep_sample = (k_keep * tb * ub)  # 实际保留块数 (B,)
        L_keep_max = L_keep_sample.max().long()
        ids_sorted, _ = torch.sort(ids_filled, dim=1)
        ids_keep = ids_sorted[:, :L_keep_max]  # (B, L_keep_max)
        
        # 安全索引处理
        is_valid = ids_keep < max_length  # 标记有效索引
        ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep))
        
        # 索引提取数据
        idx = ids_safe.unsqueeze(-1).expand(-1, -1, D)  # 扩展至特征维度
        x_masked = torch.gather(x, dim=1, index=idx)     # (B, L_keep_max, D)
        
        # 确保无效位置清零
        zero_mask = is_valid.unsqueeze(-1).expand(-1, -1, D)
        x_masked = x_masked * zero_mask.to(x_masked.dtype)
        
        # 额外处理：验证保留的低频块有效性
        actual_keep = (ids_keep < total_blocks.unsqueeze(1)) & is_valid
        if not torch.all(actual_keep == is_valid):
            print("Warning: Some preserved positions are invalid")
        
        return x_masked, mask, ids_restore, ids_keep, is_valid

    def forward_encoder(self, x, token_length, input_size, mask_ratio, mask_strategy='random'):
        # 维度转换
        # x = x[:, :-1, :, :]  # 切片处理数据维度
        # x = torch.unsqueeze(x, dim=1)

        # embed patches
        # pdb.set_trace()
        x = self.patch_embed(x)
        N, L, C = x.shape

        if mask_strategy == 'random':
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
            ids = torch.arange(self.max_length, device=x.device).unsqueeze(0).expand(N, self.max_length) 
            pad_mask_full = ids < token_length.unsqueeze(1)  # [B, L]
            # 利用 ids_keep 从 pad_mask_full 中抽出保留 token 的 padding 信息
            attn_mask = torch.gather(
                pad_mask_full, dim=1,
                index=ids_keep
            ) 
        elif mask_strategy == 'temporal':
            x, mask, ids_restore, ids_keep, is_keep = self.temporal_masking(x, input_size, mask_ratio)
            attn_mask = is_keep
        elif mask_strategy == 'freq':
            x, mask, ids_restore, ids_keep, is_keep = self.freq_masking(x, input_size, mask_ratio)
            attn_mask = is_keep
    
        # 先mask再加位置编码
        # x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        # pdb.set_trace()
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.cls_embed:
            cls_ind = 1
        else:
            cls_ind = 0
        pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
        # pdb.set_trace()
        if mask_strategy == 'temporal' or mask_strategy == 'freq':
            is_valid = ids_keep < self.max_length  # 标记有效索引 (B, L_keep_max)
            ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep)) 
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_safe.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
        else:
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )

        if self.cls_embed:
            pos_embed = torch.cat(
                [
                    self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                    pos_embed,
                ],
                1,
            )
        # pdb.set_trace()
        x = x.view([N, -1, C]) + pos_embed
        
        if self.cls_embed:
            # 创建CLS Token专用mask（始终可见）
            cls_mask = torch.ones((N, 1), dtype=attn_mask.dtype, device=attn_mask.device)
            # 拼接到原始mask前
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)  # 维度变为[N, L+1]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]
        # x: [N, x, D], mask: [N, L], ids_restore: [N, L]
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, token_length):
        # pdb.set_trace()
        N = x.shape[0]
        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, self.max_length - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, self.max_length, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, self.max_length, C])

        # create attention mask
        ids = torch.arange(self.max_length, device=x.device).unsqueeze(0).expand(N, self.max_length)  # [B, L]
        attn_mask = ids < token_length.unsqueeze(1)  

        # 扩展掩码以适应CLS Token
        if self.cls_embed:
            cls_mask = torch.ones((N, 1), dtype=attn_mask.dtype, device=attn_mask.device)
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)  # [N, L+1]

        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        
        decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, self.max_length, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, self.max_length, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def calculate_metrics_per_pixel(self, original_spectrum, reconstructed_spectrum):
        epsilon = 1e-10  # 避免除零错误

        # 计算光谱角（Spectral Angle）逐像素
        spectral_angle_per_pixel = torch.acos(torch.sum(original_spectrum * reconstructed_spectrum, dim=1) /
                                              (torch.norm(original_spectrum, dim=1) * torch.norm(reconstructed_spectrum,
                                                                                                 dim=1)+ epsilon ))#
        return spectral_angle_per_pixel

    def forward_loss(self, imgs, pred, mask):
        target1 = imgs

        if self.norm_pix_loss:
            mean = target1.mean(dim=-1, keepdim=True)
            var = target1.var(dim=-1, keepdim=True)
            target1 = (target1 - mean) / (var + 1.0e-6) ** 0.5

        loss1 = (pred - target1) ** 2  
        loss1 = loss1.mean(dim=-1) 
        mask = mask.view(loss1.shape)
        loss1 = (loss1 * mask).sum() / mask.sum()

        if not math.isfinite(loss1):
            pdb.set_trace()

        return loss1
    
    def forward_loss_v3(self, imgs, pred, mask, token_length):
        target1 = imgs
        N, L, D = imgs.shape
        col_indices = torch.arange(L, device=imgs.device).expand(N, L)
        mask_in_length = col_indices < token_length[:, None]
        bool_mask = mask.bool()
        mask_nmse = bool_mask & mask_in_length
        
        loss1 = (pred - target1) ** 2  
        loss1 = loss1.mean(dim=-1) 
        mask_nmse = mask_nmse.view(loss1.shape)
        loss1 = (loss1 * mask_nmse).sum() / mask_nmse.sum()

        if not math.isfinite(loss1):
            pdb.set_trace()

        return loss1
    
    def forward_loss_v4(self, imgs, pred, mask, token_length, input_size):
        target1 = imgs

        N, L, D = imgs.shape
        col_indices = torch.arange(L, device=imgs.device).expand(N, L)
        mask_in_length = col_indices < token_length[:, None]
        bool_mask = mask.bool()
        mask_nmse = bool_mask & mask_in_length

        loss1 = (pred - target1) ** 2  
        loss1 = loss1.mean(dim=-1) 
        mask_nmse = mask_nmse.view(loss1.shape)
        loss1 = (loss1 * mask_nmse).sum() / mask_nmse.sum()

        reconstructed = patch_recover(pred, input_size)
        doppler_loss = doppler_constraint(reconstructed)
        lambda_doppler = 0.5
        total_loss = loss1 + lambda_doppler * doppler_loss

        if not math.isfinite(total_loss):
            pdb.set_trace()

        return total_loss

    def forward(self, imgs, token_length, input_size=None, mask_ratio=0.9, mask_strategy="random"):
        latent, mask, ids_restore = self.forward_encoder(imgs, token_length, input_size, mask_ratio, mask_strategy)
        pred = self.forward_decoder(latent, ids_restore, token_length)
        # loss = self.forward_loss(imgs, pred, mask)
        loss = self.forward_loss_v3(imgs, pred, mask, token_length)
        # loss = self.forward_loss_v4(imgs, pred, mask, token_length, input_size)
        return loss, pred, mask


def mae_vit_base_patch8_96(**kwargs):
    model = MaskedAutoencoderCSI(
        img_size=96,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        pred_t_dim=12,
        t_patch_size=3,
        mask_ratio=0.90,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch8_128(**kwargs):
    model = MaskedAutoencoderCSI(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        pred_t_dim=12,
        t_patch_size=3,
        mask_ratio=0.75,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch8_csi(**kwargs):
    model = MaskedAutoencoderCSI(
        img_size=(48,64),
        in_chans=1,
        patch_size=4,
        embed_dim=768,
        num_heads=12,
        # decoder_num_heads=12,
        mlp_ratio=4,
        num_frames=16,
        pred_t_dim=16,
        t_patch_size=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model



if __name__ == '__main__':
    input = torch.rand(2, 12, 128, 128)
    model = mae_vit_base_patch8_128()
    output = model(input)
    print(output.shape)

