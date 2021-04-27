import torch
from torch import nn


class CharEncoding(nn.Module):
    def __init__(self, params={}):
        super().__init__()

        self.params = params
        self.embedding = nn.Embedding(128, params.get('embedding_dim', 64))
        self.max_line_length = params.get('max_length', 100)

        self.padding_idx = 0
        self.end_token_idx = 1

    def embed_batch(self, batch, device=None):
        batch = [self.abbreviate(s) for s in batch]

        lens = [len(s) for s in batch]
        max_len = max(lens)
        int_batch = torch.LongTensor(
            [list(s.encode('ascii')) + [self.end_token_idx] + [self.padding_idx] * (max_len - len(s))
             for s in batch])
        return self.embedding(int_batch.to(device=device)), lens

    def abbreviate(self, s):
        if len(s) > self.max_line_length:
            return s[:self.max_line_length] + '...'
        return s


# Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, base, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
