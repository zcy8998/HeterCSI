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
from torch.nn.utils.rnn import pad_sequence
from util import video_vit
from util.logging import master_print as print
from util.pos_embed import get_1d_sincos_pos_embed_from_grid


class MaskedAutoencoderCSI(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
            self,
            max_length=2048,
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
            device=None,
            **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.cls_embed = cls_embed

        self.max_length = max_length
        self.patch_embed = patch_embed(output_dim=embed_dim)
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

    def random_masking(self, x, mask_ratio, token_length):
        """
        Revised version: Perform random masking only within the effective length,
        force Padding areas to be considered 'masked' (not input to Encoder),
        and ensure Loss calculation distinguishes them.
        """
        N, L, D = x.shape
        
        # 1. Generate random noise
        noise = torch.rand(N, L, device=x.device)
        
        # 2. Logic for variable-length sequences
        # Create a mask to identify which positions are padding
        col_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(N, L)
        pad_mask = col_indices >= token_length.unsqueeze(1) # True means Padding
        
        # 3. Important: Set noise at padding positions to infinity
        # This ensures that during argsort, Padding is always sorted to the end (treated as "to be removed/masked")
        noise[pad_mask] = 1e9
        
        # 4. Sort
        # ids_shuffle: Small noise at front (Keep), large noise at back (Masked + Padding)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 5. Calculate how many tokens to keep for each sample (based on individual token_length)
        # len_keep is no longer a scalar, but a vector of shape (N,)
        len_keep = (token_length * (1 - mask_ratio)).long()
        
        # 6. Generate binary mask (0: keep, 1: remove)
        mask = torch.ones([N, L], device=x.device)
        # Since len_keep is variable, we cannot simply slice like mask[:, :len_keep] = 0
        # Need to assign using masks
        row_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(N, L)
        # Positions in the sorted indices smaller than len_keep are set to 0 (Keep)
        # Note: Comparison is against sorted order indices, i.e., "the first len_keep elements"
        keep_mask_sorted = row_indices < len_keep.unsqueeze(1) 
        
        # Restoring sorted mask to original order is tricky,
        # It is simpler to directly extract x_masked
        
        # --- Extract x_masked (tricky due to variable lengths) ---
        # To support batch processing, we usually align to a maximum kept length or allow Encoder to receive Padding
        # Here, to be compatible with existing Transformer structure, we take the maximum len_keep
        max_len_keep = len_keep.max().item()
        
        # Extract the first max_len_keep indices
        ids_keep = ids_shuffle[:, :max_len_keep]
        
        # Gather data
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # If len_keep of some samples is smaller than max_len_keep, the excess part needs to be masked (set to 0)
        # These positions were gathered but are actually padding
        valid_keep_mask = torch.arange(max_len_keep, device=x.device).unsqueeze(0) < len_keep.unsqueeze(1)
        x_masked = x_masked * valid_keep_mask.unsqueeze(-1).type_as(x_masked)
        
        # --- Generate final mask for loss ---
        # Restore mask order
        # In sorted domain, first len_keep is 0 (Keep), rest is 1 (Mask)
        mask_sorted = (~keep_mask_sorted).float()
        # Restore order
        mask = torch.gather(mask_sorted, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, ids_keep
    
    def temporal_masking(self, x, input_size, mask_ratio=0.5):
        """
        Vectorized temporal masking, simulating random masking but retaining continuous time segments.
        Parameters:
            x: Input data, (B, max_length, D)
            input_size: [Time, Frequency, Antenna] dimensions for each sample, (B, 3)
            mask_ratio: Masking ratio, default 0.5
        Returns:
            x_masked: Masked data, (B, L_keep_max, D)
            mask: Mask matrix, (B, max_length)
            ids_restore: Original indices, (B, max_length)
            ids_keep: Indices of kept positions, (B, L_keep_max)
        """
        pdb.set_trace()
        B, max_length, D = x.shape
        device = x.device
        t, k, u = input_size
        t = t.to(device)
        k = k.to(device)
        u = u.to(device)
        tb, kb, ub = t // 4, k // 4, u // 4         # Block processing (assuming 4x4 blocks)
        patches_per_t = kb * ub                     # Number of blocks per time block (B,)
        t_keep = (tb * (1 - mask_ratio)).long()     # Number of time blocks to keep (B,)
        L_keep_sample = t_keep * patches_per_t      # Actual number of blocks kept per sample (B,)
        
        # 2) Build temporal index matrix
        arange = torch.arange(max_length, device=device)  # (0 to max_length-1)
        # Calculate time block index for each position (B, max_length)
        time_idx = arange.unsqueeze(0) // patches_per_t.unsqueeze(1)

        # 3) Generate keep mask
        keep_mask = time_idx < t_keep.unsqueeze(1)  # Keep first t_keep time blocks (B, max_length)
        mask = (~keep_mask).to(torch.float32)       # Convert to float mask (B, max_length)

        # 4) Prepare index restore matrix
        ids_restore = arange.unsqueeze(0).expand(B, -1)  # (B, max_length)

        # 5) Generate keep indices (considering variable kept blocks per sample)
        filler = torch.full_like(ids_restore, max_length)  # Out-of-bounds filler value (B, max_length)
        ids_filled = torch.where(keep_mask, ids_restore, filler)  # Fill original indices for kept positions, fill max_length for others
        
        # Calculate max kept blocks (padding for sample length differences)
        L_keep_max = L_keep_sample.max().long()
        # Sort to concentrate kept indices at the front (B, max_length)
        ids_sorted, sort_indices = torch.sort(ids_filled, dim=1)
        ids_restore = torch.argsort(sort_indices, dim=1) 

        ids_keep = ids_sorted[:, :L_keep_max]  # Truncate to max kept blocks (B, L_keep_max)

        # 6) Safe index handling (avoid out-of-bounds errors)
        is_valid = ids_keep < max_length  # Mark valid indices (B, L_keep_max)
        ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep))  # Set out-of-bounds positions to 0

        # 7) Extract data using indices
        idx = ids_safe.unsqueeze(-1).expand(-1, -1, D)  # Expand to feature dimension (B, L_keep_max, D)
        x_masked = torch.gather(x, dim=1, index=idx)    # Gather data (B, L_keep_max, D)
        
        # 8) Zero out padding positions
        zero_mask = is_valid.unsqueeze(-1).expand(-1, -1, D)  # Validity mask (B, L_keep_max, D)
        x_masked = x_masked * zero_mask.to(x_masked.dtype)    # Zero out invalid positions

        return x_masked, mask, ids_restore, ids_keep, is_valid
    
    def freq_masking(self, x, input_size, mask_ratio=0.5):
        """
        Vectorized frequency masking, revised version - correctly retains low-frequency parts
        """
        B, max_length, D = x.shape
        device = x.device
        
        t, k, u = input_size
        t = t.to(device)
        k = k.to(device)
        u = u.to(device)
        # Calculate block counts (vectorized processing)
        tb = (t // 4).long()  # Time blocks
        kb = (k // 4).long()  # Freq blocks
        ub = (u // 4).long()  # Spatial blocks
        total_blocks = tb * kb * ub
        
        # ===== Core Revision: Vectorized generation of frequency block indices =====
        # Create global indices (0 to max_length-1)
        global_idx = torch.arange(max_length, device=device).unsqueeze(0)  # (1, L)
        
        # Calculate block coordinates for each position (vectorized coordinate decomposition)
        kf_tmp = global_idx // ub.unsqueeze(1)           # Intermediate value
        k_idx = kf_tmp % kb.unsqueeze(1)                 # Frequency block index (B, L)
        
        # Mark valid positions (blocks that actually exist)
        valid_mask = global_idx < total_blocks.unsqueeze(1)  # (B, L)
        
        # Calculate number of frequency blocks to keep (ensure at least 1 is kept)
        k_keep = (kb * (1 - mask_ratio)).round().long()    # (B,)
        k_keep = torch.maximum(k_keep, torch.ones_like(k_keep))
        
        # Build keep mask (retain only low-frequency blocks and valid positions)
        keep_mask = (k_idx < k_keep.unsqueeze(1)) & valid_mask  # (B, L)
        mask = (~keep_mask).to(torch.float32)  # Convert to float mask
        
        # Prepare index restore matrix
        ids_restore = torch.arange(max_length, device=device).unsqueeze(0).expand(B, -1)
        
        # Generate keep indices (considering variable kept blocks per sample)
        filler = torch.full_like(ids_restore, max_length)  # Out-of-bounds filler value
        ids_filled = torch.where(keep_mask, ids_restore, filler)
        
        # Calculate max kept blocks (padding sample length differences)
        L_keep_sample = (k_keep * tb * ub)  # Actual kept blocks (B,)
        L_keep_max = L_keep_sample.max().long()
        ids_sorted, sort_indices = torch.sort(ids_filled, dim=1)
        ids_restore = torch.argsort(sort_indices, dim=1) 

        ids_keep = ids_sorted[:, :L_keep_max]  # (B, L_keep_max)
        
        # Safe index handling
        is_valid = ids_keep < max_length  # Mark valid indices
        ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep))
        
        # Extract data using indices
        idx = ids_safe.unsqueeze(-1).expand(-1, -1, D)  # Expand to feature dimension
        x_masked = torch.gather(x, dim=1, index=idx)     # (B, L_keep_max, D)
        
        # Ensure invalid positions are zeroed out
        zero_mask = is_valid.unsqueeze(-1).expand(-1, -1, D)
        x_masked = x_masked * zero_mask.to(x_masked.dtype)
        
        # Extra processing: Verify validity of retained low-frequency blocks
        actual_keep = (ids_keep < total_blocks.unsqueeze(1)) & is_valid
        if not torch.all(actual_keep == is_valid):
            print("Warning: Some preserved positions are invalid")
        
        return x_masked, mask, ids_restore, ids_keep, is_valid

    def forward_encoder(self, x, token_length, input_size, mask_ratio, mask_strategy='random', viz=False):
        # embed patches
        pdb.set_trace()
        x = self.patch_embed(x)
        N, L, C = x.shape

        if mask_strategy == 'random':
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio, token_length)
            ids = torch.arange(L, device=x.device).unsqueeze(0).expand(N, L) 
            pad_mask_full = ids < token_length.unsqueeze(1)  # [B, L]
            # Use ids_keep to extract padding info for kept tokens from pad_mask_full
            attn_mask = torch.gather(
                pad_mask_full, dim=1,
                index=ids_keep
            ) 
        elif mask_strategy == 'temporal':
            pdb.set_trace()
            x, mask, ids_restore, ids_keep, is_keep = self.temporal_masking(x, input_size, mask_ratio)
            attn_mask = is_keep
        elif mask_strategy == 'freq':
            pdb.set_trace()
            x, mask, ids_restore, ids_keep, is_keep = self.freq_masking(x, input_size, mask_ratio)
            attn_mask = is_keep
    
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
        
        if mask_strategy == 'temporal' or mask_strategy == 'freq':
            pdb.set_trace()
            is_valid = ids_keep < L # Mark valid indices (B, L_keep_max)
            ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep)) 
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_safe.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            pos_embed = pos_embed * is_valid.unsqueeze(-1).to(pos_embed.dtype)
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
        
        x = x.view([N, -1, C]) + pos_embed
        
        if self.cls_embed:
            # Create specific mask for CLS Token (always visible)
            cls_mask = torch.ones((N, 1), dtype=attn_mask.dtype, device=attn_mask.device)
            # Concatenate before the original mask
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)  # Dimension becomes [N, L+1]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)

        if viz:
            return x # [N, max_length, embed_dim]

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]
        # x: [N, x, D], mask: [N, L], ids_restore: [N, L]
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, token_length):
        pdb.set_trace()
        N, L, _ = x.shape
        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        max_length = ids_restore.shape[1] # Get max length
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, max_length - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, max_length, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, max_length, C])

        # create attention mask
        ids = torch.arange(max_length, device=x.device).unsqueeze(0).expand(N, max_length)  # [B, L]
        attn_mask = ids < token_length.unsqueeze(1)  

        # Expand mask to accommodate CLS Token
        if self.cls_embed:
            cls_mask = torch.ones((N, 1), dtype=attn_mask.dtype, device=attn_mask.device)
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)  # [N, L+1]

        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        
        decoder_pos_embed = self.decoder_pos_embed[:, : max_length, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, max_length, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, max_length, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x
    
    def forward_loss(self, imgs, pred, mask, token_length):
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
    
    def forward(self, imgs, token_length, input_size=None, mask_ratio=0.5, mask_strategy="freq"):
        latent, mask, ids_restore = self.forward_encoder(imgs, token_length, input_size, mask_ratio, mask_strategy)
        pred = self.forward_decoder(latent, ids_restore, token_length)
        loss = self.forward_loss(imgs, pred, mask, token_length)
        return loss, pred, mask


def mae_vit_base_csi(**kwargs):
    model = MaskedAutoencoderCSI(
        embed_dim=512,
        depth=6,
        num_heads=8,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch8_csi(**kwargs):
    model = MaskedAutoencoderCSI(
        embed_dim=768,
        num_heads=12,
        depth=12,
        mlp_ratio=4,
        num_frames=16,
        t_patch_size=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def mae_vit_csi(**kwargs):
    model = MaskedAutoencoderCSI(
        embed_dim=768,
        num_heads=12,
        depth=8,
        mlp_ratio=4,
        decoder_num_heads=16,
        t_patch_size=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


if __name__ == '__main__':
    input = torch.rand(2, 12, 128, 128)
    model = mae_vit_base_patch8_csi()
    output = model(input)
    print(output.shape)