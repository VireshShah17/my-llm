import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    """
        Batch-aware attention where token i can only see tokens <= i.
    """

    mask: torch.Tensor

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # buffer: moves with the model to GPU/CPU but is not a trainable param
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1).bool(),)


    def forward(self, x):
        num_tokens = x.shape[1]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)           # [b, T, T]
        # fill future positions with -inf so softmax gives them weight 0
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values
