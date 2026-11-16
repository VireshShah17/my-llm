import pytest
import torch
import torch.nn as nn
from src.attention import (INPUTS, CausalAttention,)


D_IN = 3
D_OUT = 2
T = 6


def make_attn(dropout = 0.0, context_length = T, qkv_bias = False, seed = 123):
    torch.manual_seed(seed)
    return CausalAttention(D_IN, D_OUT, context_length, dropout, qkv_bias = qkv_bias)


def make_batch(batch_size = 2):
    return torch.stack([INPUTS] * batch_size, dim = 0)      # [b, 6, 3]


def reference_attention(m, x):
    """
        Plain re-implementation: returns (context, weights) with a lower-triangular mask.
    """
    queries, keys, values = m.W_query(x), m.W_key(x), m.W_value(x)
    scores = queries @ keys.transpose(1, 2)
    n = x.shape[1]
    future = torch.triu(torch.ones(n, n), diagonal = 1).bool()
    scores = scores.masked_fill(future, -torch.inf)
    weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim = -1)

    return weights @ values, weights


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_uses_linear_layers_with_correct_sizes():
    m = make_attn()

    for layer in (m.W_query, m.W_key, m.W_value):
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == D_IN
        assert layer.out_features == D_OUT


def test_no_bias_by_default_and_bias_flag_adds_parameters():
    assert len(list(make_attn().parameters())) == 3
    assert len(list(make_attn(qkv_bias = True).parameters())) == 6


def test_mask_has_context_length_shape_and_is_strictly_upper_triangular():
    m = make_attn(context_length = 8)
    assert m.mask.shape == (8, 8)
    expected = torch.triu(torch.ones(8, 8), diagonal = 1).bool()
    assert torch.equal(m.mask.bool(), expected)
    assert not m.mask.diagonal().any()           # a token may attend to itself


def test_mask_is_a_buffer_not_a_trainable_parameter():
    m = make_attn()
    assert "mask" in dict(m.named_buffers())
    assert "mask" not in dict(m.named_parameters())
    assert "mask" in m.state_dict()


def test_state_dict_roundtrip_preserves_output():
    src, dst = make_attn(seed = 1), make_attn(seed = 2)
    dst.load_state_dict(src.state_dict())
    x = make_batch()
    assert torch.allclose(src(x), dst(x))


# ---------------------------------------------------------------------------
# Forward : shapes and values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_output_shape(batch_size):
    out = make_attn()(make_batch(batch_size))
    assert out.shape == (batch_size, T, D_OUT)


def test_output_is_finite():
    assert torch.isfinite(make_attn()(make_batch())).all()


def test_matches_reference_implementation():
    m = make_attn()
    x = make_batch()
    expected, _ = reference_attention(m, x)
    assert torch.allclose(m(x), expected, atol = 1e-6)


def test_reference_weights_are_lower_triangular_and_rows_sum_to_one():
    m = make_attn()
    _, w = reference_attention(m, make_batch())
    assert w.shape == (2, T, T)
    assert torch.allclose(w.sum(dim = -1), torch.ones(2, T), atol = 1e-6)
    assert torch.allclose(torch.triu(w, diagonal = 1), torch.zeros(2, T, T))
    allowed = torch.tril(torch.ones(T, T)).bool()
    assert (w[:, allowed] > 0).all()             # every allowed position gets weight


def test_first_token_output_equals_its_own_value_vector():
    """
        Token 0 can only see itself -> softmax over one score = 1.
    """
    m = make_attn()
    x = make_batch()
    assert torch.allclose(m(x)[:, 0], m.W_value(x)[:, 0], atol = 1e-6)


def test_batch_items_are_processed_independently():
    m = make_attn()
    a = INPUTS.unsqueeze(0)
    b = (INPUTS.flip(0) * 0.5).unsqueeze(0)
    together = m(torch.cat([a, b], dim = 0))
    assert torch.allclose(together[0], m(a)[0], atol = 1e-6)
    assert torch.allclose(together[1], m(b)[0], atol = 1e-6)


# ---------------------------------------------------------------------------
# Causality (the whole point of this layer)
# ---------------------------------------------------------------------------
def test_changing_last_token_does_not_change_earlier_outputs():
    m = make_attn()
    x = make_batch()
    changed = x.clone()
    changed[:, -1] += 1.0
    assert torch.allclose(m(changed)[:, :-1], m(x)[:, :-1], atol = 1e-6)


@pytest.mark.parametrize("k", [0, 2, 4])
def test_changing_token_k_leaves_outputs_before_k_untouched_but_affects_k(k):
    m = make_attn()
    x = make_batch()
    changed = x.clone()
    changed[:, k] += 1.0
    out_old, out_new = m(x), m(changed)

    if k > 0:
        assert torch.allclose(out_new[:, :k], out_old[:, :k], atol = 1e-6)
    assert not torch.allclose(out_new[:, k], out_old[:, k])


def test_changing_an_early_token_does_affect_later_outputs():
    m = make_attn()
    x = make_batch()
    changed = x.clone()
    changed[:, 0] += 1.0
    assert not torch.allclose(m(changed)[:, -1], m(x)[:, -1])


# ---------------------------------------------------------------------------
# Sequence length vs context_length
# ---------------------------------------------------------------------------
def test_shorter_sequence_than_context_length_works():
    m = make_attn(context_length = 10)
    out = m(make_batch()[:, :4])
    assert out.shape == (2, 4, D_OUT)


def test_shorter_sequence_equals_prefix_of_full_sequence_output():
    """
        Causal => output for the first n tokens must not depend on later tokens.
    """
    m = make_attn()
    x = make_batch()
    assert torch.allclose(m(x[:, :3]), m(x)[:, :3], atol = 1e-6)


def test_sequence_longer_than_context_length_raises():
    m = make_attn(context_length = 4)

    with pytest.raises(RuntimeError):
        m(make_batch())                          # 6 tokens > context_length 4


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------
def test_dropout_module_uses_given_probability():
    assert make_attn(dropout = 0.3).dropout.p == 0.3


def test_eval_mode_ignores_dropout():
    x = make_batch()
    with_drop = make_attn(dropout = 0.5).eval()
    no_drop = make_attn(dropout = 0.0).eval()
    assert torch.allclose(with_drop(x), no_drop(x), atol = 1e-6)


def test_train_mode_dropout_changes_output_and_is_random():
    m = make_attn(dropout = 0.5).train()
    x = make_batch()
    torch.manual_seed(0)
    first = m(x)
    torch.manual_seed(1)
    second = m(x)
    assert not torch.allclose(first, second)
    m.eval()
    assert not torch.allclose(first, m(x))


def test_train_mode_with_zero_dropout_is_deterministic():
    m = make_attn(dropout = 0.0).train()
    x = make_batch()
    assert torch.allclose(m(x), m(x))


def test_dropout_keeps_output_finite():
    m = make_attn(dropout = 0.5).train()
    assert torch.isfinite(m(make_batch())).all()


# ---------------------------------------------------------------------------
# Training behaviour
# ---------------------------------------------------------------------------
def test_gradients_flow_to_all_parameters():
    m = make_attn(qkv_bias = True)
    m(make_batch()).sum().backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all()


def test_gradient_of_first_output_ignores_later_tokens():
    """
        d out[:, 0] / d x[:, 1:] must be zero (future tokens are hidden).
    """
    m = make_attn()
    x = make_batch().clone().requires_grad_(True)
    m(x)[:, 0].sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad[:, 1:], torch.zeros_like(x.grad[:, 1:]))
    assert x.grad[:, 0].abs().sum() > 0


def test_forward_does_not_modify_the_input():
    m = make_attn()
    x = make_batch()
    before = x.clone()
    m(x)
    assert torch.equal(x, before)
