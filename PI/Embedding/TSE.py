import torch
import torch.nn as nn
from datetime import datetime


class TimestampEmbedding(nn.Module):
    def __init__(self, d_model: int, max_year=2050, min_year=2000):
        super(TimestampEmbedding, self).__init__()

        self.year_embedding = nn.Embedding(max_year - min_year + 1, d_model // 4)
        self.month_embedding = nn.Embedding(12, d_model // 4)
        self.day_of_week_embedding = nn.Embedding(7, d_model // 4)
        self.hour_embedding = nn.Embedding(24, d_model // 4)

        for embedding in [self.year_embedding, self.month_embedding, self.day_of_week_embedding, self.hour_embedding]:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, timestamps):
        """
        Args:
            timestamps: Tensor of shape [batch_size] containing timestamps in Unix timestamp format.
        Returns:
            Tensor of shape [batch_size, d_model] with the concatenated embeddings.
        """

        dt = [datetime.fromtimestamp(ts.item()) for ts in timestamps]
        years = torch.tensor([d.year for d in dt], device=timestamps.device) - 2000
        months = torch.tensor([d.month for d in dt], device=timestamps.device) - 1
        days_of_week = torch.tensor([d.weekday() for d in dt], device=timestamps.device)
        hours = torch.tensor([d.hour for d in dt], device=timestamps.device)

        year_emb = self.year_embedding(years)
        month_emb = self.month_embedding(months)
        day_of_week_emb = self.day_of_week_embedding(days_of_week)
        hour_emb = self.hour_embedding(hours)

        embeddings = torch.cat((year_emb, month_emb, day_of_week_emb, hour_emb), dim=-1)

        return embeddings