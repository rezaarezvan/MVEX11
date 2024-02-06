import math
import torch
import torch.nn as nn


# Mapping words to vectors
class inputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Maps words to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

    # uhh xd
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        # Tensor (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    # Add to every word in sentence
    def forward(self, x):
        x = x + (self.pe[:, x.shape[1], :]).requires_grad(False)  # Static
        return self.dropout(x)


class LayerNormalisation(nn.Module):
    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        # Avoid division by 0
        self.eps = eps
        # nn.Parameter(torch.ones(1)) -> learnable parameter
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    # \hat{x}  =\frac{x_j - \mu_j}{\sqrt{\mu^2_j+ \epsilon}}
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, dff)  # First layer
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff, d_model)  # Second layer

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, dff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, 'd_model is not divisible my h'
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Replace 0 by -1e9 when masking (prohibit interaction with words not yet reached(-\infty))
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    # Mask is done to prohibit interaction with words not yet reached (upper half of matrix QK^T = -\infty)
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split up into different heads
        # Hocus pocus
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        # d_k  = d_model // h
        # Add together all the separate heads
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation()

    # Add the multi-headed output (sublayer) and regular output (x)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EnconderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.feed_forward_block = feed_forward_block

    # src_mask to remove paddings
    def forward(self, x, src_mask):
        # (x,x,x, src_mask) \iff (q,k,v, mask)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
