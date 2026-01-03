import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from einops import rearrange


class CSIBERT(nn.Module):
    def __init__(self, feature_dim, num_hidden_layers=6, num_attention_heads=8):
        super(CSIBERT, self).__init__()
        # 使用较轻量级的配置，你可以根据显存大小调整 layers 和 heads
        self.config = BertConfig(
            hidden_size=512,              # 隐藏层维度，可调 (例如 768)
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=1024,       # 前馈网络维度
            max_position_embeddings=512   # 足够覆盖子载波数 K
        )
        self.bert = BertModel(self.config)

        # 嵌入层
        # time_embedding 在这里实际上是 "Frequency/Subcarrier Embedding"
        self.time_embedding = nn.Embedding(512, self.config.hidden_size)
        self.feature_embedding = nn.Linear(feature_dim, self.config.hidden_size)

        # 输出层：重建原始特征
        self.output_layer = nn.Linear(self.config.hidden_size, feature_dim)

    def forward(self, inputs, attention_mask=None):
        # inputs: (batch_size, seq_len, feature_dim)
        batch_size, sequence_length, feature_dim = inputs.shape
        device = inputs.device

        # 1. 生成位置/频率嵌入 (Subcarrier Positional Embedding)
        # 生成 [0, 1, ..., K-1]
        freq_indices = torch.arange(sequence_length, device=device).unsqueeze(0) 
        pos_embeds = self.time_embedding(freq_indices).expand(batch_size, -1, -1)

        # 2. 生成特征嵌入
        feature_embeds = self.feature_embedding(inputs)

        # 3. 融合
        combined_embeds = pos_embeds + feature_embeds

        # 4. BERT Forward
        # 注意：我们跳过了 tokenizer，直接输入 embeddings
        outputs = self.bert(inputs_embeds=combined_embeds,
                            attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state

        # 5. 输出重建
        predictions = self.output_layer(hidden_states)
        
        return predictions