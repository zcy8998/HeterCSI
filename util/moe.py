import torch
import torch.nn as nn
import torch.nn.functional as F

from util.video_vit import Attention

# class MoE_MLP(nn.Module):
#     def __init__(self, dim, mlp_ratio=4.0, num_experts=4, top_k=2, dropout=0.0):
#         super().__init__()
#         self.num_experts = num_experts
#         self.top_k = top_k
#         hidden_features = int(dim * mlp_ratio)

#         # 1. 门控网络 (Router)
#         self.gate = nn.Linear(dim, num_experts)
        
#         # 2. 专家网络 (这里简单的使用ModuleList，你可以根据显存情况调整数量)
#         # 每个专家就是一个标准的 MLP
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, hidden_features),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_features, dim),
#                 nn.Dropout(dropout)
#             ) for _ in range(num_experts)
#         ])

#     def forward(self, x):
#         # x: [Batch, Length, Dim]
#         B, L, D = x.shape
#         x_flat = x.view(-1, D)

#         # 1. 计算 Router Logits
#         gate_logits = self.gate(x_flat) # [N, num_experts]
        
#         # 2. 选出 Top-K 专家
#         weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
#         weights = F.softmax(weights, dim=-1) # [N, k]
        
#         # 3. 计算辅助 Loss (Load Balancing Loss) 的一部分：Importance
#         # 我们把它存起来，外部可以访问，或者直接这里不做处理，简化实现
#         self.last_gate_logits = gate_logits # 留给 loss 计算用

#         # 4. 路由计算
#         final_output = torch.zeros_like(x_flat)
        
#         # 简单的循环实现（适用于 expert 数量少的情况，如 4-8 个）
#         for k in range(self.top_k):
#             expert_idx_k = indices[:, k]
#             expert_weight_k = weights[:, k].unsqueeze(1)
            
#             for i in range(self.num_experts):
#                 # 找到分配给专家 i 的 token mask
#                 mask = (expert_idx_k == i)
#                 if mask.any():
#                     inp = x_flat[mask]
#                     out = self.experts[i](inp)
#                     final_output[mask] += out * expert_weight_k[mask]
        
#         return final_output.view(B, L, D)

class MoE_MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, num_experts=4, top_k=2, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_features = int(dim * mlp_ratio)

        self.gate = nn.Linear(dim, num_experts)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_features),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [Batch, Length, Dim]
        B, L, D = x.shape
        x_flat = x.view(-1, D)

        # 1. 计算 Router Logits
        gate_logits = self.gate(x_flat) 
        self.last_gate_logits = gate_logits # 保存用于 Loss 计算
        
        # 2. 选出 Top-K 专家
        # weights: [N, k], indices: [N, k]
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # 3. 初始化输出
        final_output = torch.zeros_like(x_flat)
        
        # 4. 优化的调度策略 (DDP Friendly)
        # 不再循环 k，而是循环 Experts。
        # 确保每个 Expert 在一次 Forward 中只被调用一次。
        
        for i in range(self.num_experts):
            # 查找哪些 token 选中了专家 i (无论是在 top-1 还是 top-2 位置)
            # indices shape: [N, k]
            # selection_mask shape: [N, k] (布尔值)
            selection_mask = (indices == i)
            
            # 只要有任意一个 token 选中了这个专家
            if selection_mask.any():
                # 计算选中了专家 i 的 token 在 batch 中的索引
                # dim=-1 做或运算：只要 token n 的 top-k 里包含专家 i，这一行就是 True
                batch_indices_mask = selection_mask.any(dim=-1) # shape: [N]
                
                # 选出输入数据 (只选需要计算的 token)
                # inp shape: [M, D], M 是选中该专家的 token 数量
                inp = x_flat[batch_indices_mask]
                
                # *** 核心修改：每个专家只运行一次 ***
                out = self.experts[i](inp)
                
                # 准备加权系数
                # 即使一个 token 在 top-k 里多次选中同一个专家(极少见)，这里 sum 也能处理
                # weights: [N, k], selection_mask: [N, k]
                # w shape: [N] -> [M]
                w = (weights * selection_mask.float()).sum(dim=-1)
                w = w[batch_indices_mask].unsqueeze(-1) # [M, 1]
                
                # 将结果加回到 final_output
                # 这种索引加法操作 (index_add) 在 PyTorch 中是自动处理梯度的
                final_output[batch_indices_mask] += out * w
        
        return final_output.view(B, L, D)
    

# 定义一个 MoE Block，用来替换 video_vit.Block_v2
class MoEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, num_experts=4, top_k=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = norm_layer(dim)
        
        # *** 关键修改：使用 MoE_MLP ***
        self.mlp = MoE_MLP(dim, mlp_ratio=mlp_ratio, num_experts=num_experts, top_k=top_k)

        # 简单的 DropPath 占位（如果你的 video_vit 里有，请确保导入）
        self.drop_path = nn.Identity() 

    def forward(self, x, attn_mask=None):
        # 注意：这里假设 video_vit.Attention 接受 attn_mask
        # 如果原来的 Block_v2 逻辑不同，请参照原代码调整
        x = x + self.drop_path(self.attn(self.norm1(x))) # 如果 Attn 需要 mask，加进去
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x