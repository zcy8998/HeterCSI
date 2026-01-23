import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # 5000,1

        div_term = (torch.arange(0, d_model, 2).float()  # 256
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 1,5000,512
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (1, 16, 768)
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # pdb.set_trace()
        # (b, len, 96)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # print(self.tokenConv.weight)
        # print(self.tokenConv.weight.grad)
        # (1024, 16, 768)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        # pdb.set_trace()
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        # pdb.set_trace()
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()


        # 将输入的数值特征映射到高维空间
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            # 2,25,512   1,25,512
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_v2(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_v2, self).__init__()


        # 将输入的数值特征映射到高维空间
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.tc_embedding = TcEmbedding(beta_norm=1, eta_threshold=0.3)  # TcEmbedding 实现
        self.tc_proj = torch.nn.Linear(1, d_model) 
        # 信道相干时间

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):    
        tc = self.tc_embedding(x)  # 计算 Tc 向量 (batch, seq)
        # 投影Tc (添加一个维度使其变为(batch, seq, 1))
        tc_embed = self.tc_proj(tc.unsqueeze(-1))  

        x = self.value_embedding(x) + self.position_embedding(x) + tc_embed
        return self.dropout(x)


class TcEmbedding(torch.nn.Module):
    def __init__(self, beta_norm: int = 1, eta_threshold: float = 0.3):
        """
        TcEmbedding 实现
        :param beta_norm: 范数类型 (1 或 2)
        :param eta_threshold: 变化率阈值
        """
        super().__init__()
        self.beta_norm = beta_norm
        self.eta_threshold = eta_threshold
        self.eps = 1e-8  # 避免除零错误

    def compute_tc_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 Tc 向量 (相对相干时间序列)
        :param x: CSI 输入张量 (batch_size, seq_len, N_s)
        :return: Tc 向量 (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算幅度 (复数 → 实数)
        x_mag = torch.abs(x)  # (batch_size, seq_len, N_s)
        
        # 计算每个时刻的范数 ||x_t||_β
        if self.beta_norm == 1:
            norms = torch.norm(x_mag, p=1, dim=-1)  # L1 范数 (batch_size, seq_len)
        else:
            norms = torch.norm(x_mag, p=2, dim=-1)  # L2 范数 (batch_size, seq_len)
        
        # 初始化 Tc 向量 (全零)
        tc_vector = torch.zeros_like(norms, dtype=torch.float32)
        
        # 遍历每个样本和每个时间步
        for b in range(batch_size):
            # print(b)
            for t in range(1, seq_len):  # t=0 无法回溯，保持为0
                # 尝试所有可能的回溯步长 τ (从1到t)
                for tau in range(1, t+1):
                    # 计算变化率 η_t = 1 - ||x_{t-τ}||_β / ||x_t||_β
                    norm_ratio = norms[b, t-tau] / (norms[b, t] + self.eps)
                    eta_t = 1.0 - norm_ratio
                    
                    # 当首次超过阈值时记录 τ
                    if eta_t > self.eta_threshold:
                        tc_vector[b, t] = tau
                        break
                # 如果所有 τ 都未超过阈值，保持0 (论文要求)
        return tc_vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: CSI 输入 (batch_size, seq_len, N_s)
        :return: TcEmbedding 向量 (batch_size, seq_len)
        """
        # pdb.set_trace()
        return self.compute_tc_vector(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        # pdb.set_trace()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()
        
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, config, prefix='attn', freq_distribution=None, n_channels=None, init_floor_freq=None, init_upper_freq=None):
        super().__init__()
        self.config = config
        self.prefix = prefix
        
        self.dim = self.config.d_model // self.config.n_heads
        # 频率分布类型（如线性、常量、高斯分布等）。
        self.freq_distribution = freq_distribution if freq_distribution is not None else self.config.rope_init_distribution
        self.init_floor_freq = init_floor_freq if init_floor_freq is not None else self.config.rope_init_floor_freq
        self.init_upper_freq = init_upper_freq if init_upper_freq is not None else self.config.rope_init_upper_freq

        if n_channels is not None:
            self.n_channels = n_channels
        elif self.prefix == "attn" and self.config.rope_learnable and self.config.rope_no_repetition:
            self.n_channels = self.config.n_heads
        else:
            self.n_channels = 1

        self.inv_freq = self.get_inv_freq(self.dim, device=get_device(config))
        
    def get_inv_freq(self, dim, device=None):
        # pdb.set_trace()
        if self.freq_distribution == "constant": # self.config.rope_no_rotary:
            torch.ones_like(torch.arange(0, dim, 2, device=device, dtype=torch.float))
        elif self.freq_distribution == "linear": # self.config.rope_linear:
            inv_freq = 2*torch.pi/self.config.max_sequence_length * torch.arange(0, dim, 2, device=device, dtype=torch.float) 
            inv_freq[inv_freq > self.clamp_upper_ratio * torch.pi] = self.clamp_upper_value
            # 翻转，保持频率递减
            inv_freq = inv_freq.flip(0)
        elif self.freq_distribution == "uniform": # self.config.rope_uniform:
            inv_freq = 1.0 * torch.rand(dim//2, device=device, dtype=torch.float)
            
        elif self.freq_distribution == "gaussian": # self.config.rope_gaussian:
            inv_freq = torch.randn(dim//2, device=device, dtype=torch.float).abs()
            inv_freq = inv_freq / inv_freq.max()
        else:
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
        
        inv_freq = self.init_floor_freq + inv_freq * (self.init_upper_freq - self.init_floor_freq)
        
        # if self.config.fourier and self.config.fourier_ignore_zero:
        #     inv_freq = inv_freq[inv_freq != 0.0]
        
        if self.prefix == "embed":
            inv_freq = inv_freq # shape: (dim//2, )
        elif self.prefix == "attn":
            if self.config.rope_learnable and self.config.rope_no_repetition:
                inv_freq = inv_freq.repeat(self.n_channels, 1) # shape: (n_heads, dim//2)
            else:
                inv_freq = inv_freq[None, :] # shape: (1, dim//2)
        else:
            raise ValueError(f"Unsupported prefix: {self.prefix}")
        
        return inv_freq
    
    def get_rotary_embedding(self, seq_len, device):
        with torch.autocast(device.type, enabled=False):
            # pdb.set_trace()
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            if self.prefix == "embed":
                freqs = torch.einsum("t, d -> td", seq, self.inv_freq) # shape: (seq_len, dim//2)
            elif self.prefix == "attn":
                freqs = torch.einsum("t, hd -> htd", seq, self.inv_freq) # shape: (1 or n_heads, seq_len, dim//2)
            else:
                raise ValueError(f"Unsupported prefix: {self.prefix}")

            if self.config.pe_type == 'fope':
                positions = freqs.unsqueeze(0)
            elif self.config.pe_type == 'rope':
                positions = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)

            return positions.sin(), positions.cos()

    # 将输入张量的最后一维分成两部分，交换顺序并对其中一部分取负。
    def rotate_half(self, x):
        # pdb.set_trace()
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)

        x1, x2 = x.unbind(dim=-2)

        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t):
        # pdb.set_trace()
        return ((t * pos_cos) - (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, x, all_len):
        if self.config.rope_full_precision:
            x_ = x.float()
        else:
            x_ = x

        with torch.autocast(x.device.type, enabled=False):
            x_len = x_.shape[-2]
            # pdb.set_trace()
            # (1, 1, seq_len, dim//2) 
            pos_sin, pos_cos = self.get_rotary_embedding(all_len, x_.device)
            pos_sin = pos_sin.type_as(x_)
            pos_cos = pos_cos.type_as(x_)

            if self.prefix == "embed":
                x_ = self.apply_rotary_pos_emb(
                    pos_sin[:, all_len - x_len : x_len, :], 
                    pos_cos[:, all_len - x_len : x_len, :], 
                    x_,
                )
            elif self.prefix == "attn":
                x_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, all_len - x_len : all_len, :], 
                    pos_cos[:, :, all_len - x_len : all_len, :], 
                    x_,
                )
            else:
                raise ValueError(f"Unsupported prefix: {self.prefix}")

        return x_.type_as(x)

    
def get_device(pe_config):
    if pe_config.device is not None:
        return pe_config.device