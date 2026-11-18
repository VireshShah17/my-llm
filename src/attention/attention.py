"""
    Chapter 3 - Coding attention mechanisms
    =======================================
        3.3  Simple self-attention (no trainable weights)
        3.4  Self-attention with trainable weights (query / key / value)
        3.5  Causal attention (masking future tokens + dropout)
        3.6  Multi-head attention (wrapper version, then the efficient version)

    Requirements:  pip install torch
"""

import torch
import torch.nn as nn


INPUTS = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],   
     [0.57, 0.85, 0.64],   
     [0.22, 0.58, 0.33],   
     [0.77, 0.25, 0.10],   
     [0.05, 0.80, 0.55]]   
)


def context_vec_loop(inputs: torch.Tensor, query_idx: int) -> torch.Tensor:
    """
        Loop version for ONE query token.
        1. scores[i] = torch.dot(inputs[i], query)   for every i
        2. weights   = torch.softmax(scores, dim=0)
        3. context   = sum_i weights[i] * inputs[i]
        Returns shape [d].
    """
    query = inputs[query_idx]

    # Attention score = dot product of every input with the query
    scores = torch.empty(inputs.shape[0])

    for i, x_i in enumerate(inputs):
        scores[i] = torch.dot(x_i, query)

    # Normalise the weight with softmax so that they sum upto 1
    weights = torch.softmax(scores, dim = 0)

    # Context vector = weighted sum of all inputs
    context = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context += weights[i] * x_i

    return context


def simple_attention_weights(inputs: torch.Tensor) -> torch.Tensor:
    """
        Returns [T, T]: softmax over the last dim of inputs @ inputs.T
    """
    scores = inputs @ inputs.T
    return torch.softmax(scores, dim = -1)


def simple_attention(inputs: torch.Tensor) -> torch.Tensor:
    """
        Returns [T, d] context vectors: weights @ inputs
    """
    return simple_attention_weights(inputs) @ inputs


# ---------------------------------------------------------------------------
# 3.4  Self-attention with trainable weights
# ---------------------------------------------------------------------------
class SelfAttention_v1(nn.Module):
    """
        Scaled dot-product attention using raw nn.Parameter matrices.
    """

    def __init__(self, d_in, d_out) -> None:
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))


    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        # divide by sqrt(d_k) so softmax does not saturate (-> tiny gradients)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        
        return attn_weights @ values


class SelfAttention_v2(nn.Module):
    """
        Same as v1 but with nn.Linear (better weight init, optional bias).
    """

    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)


    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)

        return attn_weights @ values
