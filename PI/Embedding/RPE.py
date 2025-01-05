import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, "Embedding dimension must be even to split into two equal parts."

        freqs = torch.pow(base, -torch.arange(0, d_model, 2).float() / d_model)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """
        Args:
            x: Token embeddings of shape [seq_len, batch_size, embedding_dim]

        """
        seq_len = x.size(0)
        position = torch.arange(seq_len, device=x.device).type_as(self.freqs)
        freqs = position[:, None] * self.freqs[None, :]  # [seq_len, d_model // 2]
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # [seq_len, d_model]

        x_rot = x.view(seq_len, -1, self.d_model // 2, 2)
        x_rot = torch.stack((-x_rot[:, :, :, 1], x_rot[:, :, :, 0]), dim=-1)
        x_rot = x_rot.view(seq_len, -1, self.d_model)

        x = x * emb.cos() + x_rot * emb.sin()
        return x