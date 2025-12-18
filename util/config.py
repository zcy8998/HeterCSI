class Config:
    def __init__(self, d_model, n_heads, fope=False, rope_theta=10000, max_sequence_length=512, device='cpu', rope_fourier_init_norm_gain=0.3, rope_full_precision=False, rope_init_distribution="exponential", pe_type="fope", pe_layers="all"):
        self.d_model = d_model  # 模型的隐藏维度
        self.n_heads = n_heads  # 注意力头的数量
        self.fope = fope  # 是否启用 Fourier Position Embedding
        self.rope_theta = rope_theta  # Rotary Position Embedding 的 theta 参数
        self.max_sequence_length = max_sequence_length  # 最大序列长度
        self.device = device  # 设备类型（如 'cpu' 或 'cuda'）
        self.rope_fourier_init_norm_gain = rope_fourier_init_norm_gain  # Fourier 初始化增益
        self.rope_full_precision = rope_full_precision
        self.rope_init_distribution = rope_init_distribution
        self.rope_learnable = False  # 是否启用可学习的 Fourier Position Embedding
        self.rope_no_repetition = False  # 是否启用无重复的 Fourier Position Embedding``
        self.rope_init_floor_freq: float = 0.0
        self.rope_init_upper_freq: float = 1.0
        self.fourier_separate_basis = True
        self.fourier_norm = False
        self.fourier_separate_head = True
        self.pe_type = pe_type
        self.pe_layers = pe_layers