import torch
import torch.nn as nn
import math


class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_relative_position: int, d_model: int):
        super(RelativePositionalEncoding, self).__init__()


        vocab_size = max_relative_position * 2 + 1
        self.embeddings_table = nn.Parameter(torch.randn(vocab_size, d_model))


        self.max_relative_position = max_relative_position
        self.register_buffer('relative_positions_list', self._generate_relative_positions(max_relative_position))

    def _generate_relative_positions(self, max_relative_position):

        distance_mat = torch.arange(-max_relative_position, max_relative_position + 1).unsqueeze(0)
        return distance_mat

    def forward(self, length_q: int, length_k: int):
        """
        Args:
            length_q: length of Query
            length_k: length of Key
        Returns:
            relative positional embedding, size [length_q, length_k, d_model]
        """

        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]  # [length_q, length_k]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position

        final_mat = final_mat.to(self.relative_positions_list.device)
        embeddings = self.embeddings_table[final_mat].to(self.embeddings_table.device)  # [length_q, length_k, d_model]
        return embeddings