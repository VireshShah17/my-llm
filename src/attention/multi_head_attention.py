import torch
import torch.nn as nn
from .causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    """
        Naive version: run several CausalAttention heads and concatenate the results.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )


    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim = -1)


class MultiHeadAttention(nn.Module):
    """
        Efficient version: one big Q/K/V projection, split into heads with view/transpose.
    """

    mask: torch.Tensor

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)         # combines the heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal = 1).bool(),
        )

    
    def forward(self, x):
        b, num_tokens, _ = x.shape

        # [b, T, d_out] -> [b, T, heads, head_dim] -> [b, heads, T, head_dim]
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)                    # [b, heads, T, T]
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)

        # scale by sqrt(head_dim), NOT sqrt(d_out)
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        # [b, heads, T, head_dim] -> [b, T, heads, head_dim] -> [b, T, d_out]
        context = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        return self.out_proj(context)
