import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from einops import rearrange


class CSIBERT(nn.Module):
    def __init__(self, feature_dim, num_hidden_layers=6, num_attention_heads=8):
        super(CSIBERT, self).__init__()
        # Use a lightweight configuration; you can adjust layers and heads based on VRAM size
        self.config = BertConfig(
            hidden_size=512,              # Hidden layer dimension, adjustable (e.g., 768)
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=1024,       # Feed-forward network dimension
            max_position_embeddings=512   # Sufficient to cover the number of subcarriers K
        )
        self.bert = BertModel(self.config)

        # Embedding layers
        # time_embedding is actually "Frequency/Subcarrier Embedding" here
        self.time_embedding = nn.Embedding(512, self.config.hidden_size)
        self.feature_embedding = nn.Linear(feature_dim, self.config.hidden_size)

        # Output layer: Reconstruct original features
        self.output_layer = nn.Linear(self.config.hidden_size, feature_dim)

    def forward(self, inputs, attention_mask=None):
        # inputs: (batch_size, seq_len, feature_dim)
        batch_size, sequence_length, feature_dim = inputs.shape
        device = inputs.device

        # 1. Generate positional/frequency embeddings (Subcarrier Positional Embedding)
        # Generate [0, 1, ..., K-1]
        freq_indices = torch.arange(sequence_length, device=device).unsqueeze(0) 
        pos_embeds = self.time_embedding(freq_indices).expand(batch_size, -1, -1)

        # 2. Generate feature embeddings
        feature_embeds = self.feature_embedding(inputs)

        # 3. Combine/Fuse
        combined_embeds = pos_embeds + feature_embeds

        # 4. BERT Forward
        # Note: We skip the tokenizer and input embeddings directly
        outputs = self.bert(inputs_embeds=combined_embeds,
                            attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state

        # 5. Output reconstruction
        predictions = self.output_layer(hidden_states)
        
        return predictions