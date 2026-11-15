import pytest
import torch
import torch.nn as nn
from src.attention import (
    INPUTS,
    SelfAttention_v1,
    SelfAttention_v2,
)


D_IN = 3
D_OUT = 2
T = 6


def make_v1(seed = 123):
    torch.manual_seed(seed)
    return SelfAttention_v1(D_IN, D_OUT)


def make_v2(seed = 789, qkv_bias = False):
    torch.manual_seed(seed)
    return SelfAttention_v2(D_IN, D_OUT, qkv_bias = qkv_bias)


def manual_attention(x, W_q, W_k, W_v):
    """
        Reference implementation: softmax(QK^T / sqrt(d_k)) V, using plain matrices.
    """
    queries, keys, values = x @ W_q, x @ W_k, x @ W_v
    scores = queries @ keys.T
    weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim = -1)
    return weights @ values, weights


def copy_v2_into_v1(v1, v2):
    """
        nn.Linear stores weights as [d_out, d_in]; v1 stores [d_in, d_out] -> transpose.
    """
    v1.W_query.data = v2.W_query.weight.T.detach().clone()
    v1.W_key.data = v2.W_key.weight.T.detach().clone()
    v1.W_value.data = v2.W_value.weight.T.detach().clone()


# ---------------------------------------------------------------------------
# SelfAttention_v1 : structure
# ---------------------------------------------------------------------------
def test_v1_weight_shapes():
    m = make_v1()
    assert m.W_query.shape == (D_IN, D_OUT)
    assert m.W_key.shape == (D_IN, D_OUT)
    assert m.W_value.shape == (D_IN, D_OUT)


def test_v1_has_three_trainable_parameters():
    m = make_v1()
    params = list(m.parameters())
    assert len(params) == 3
    assert all(isinstance(p, nn.Parameter) and p.requires_grad for p in params)


def test_v1_init_is_uniform_in_zero_one():
    m = make_v1()

    for w in (m.W_query, m.W_key, m.W_value):
        assert (w >= 0).all() and (w < 1).all()


# ---------------------------------------------------------------------------
# SelfAttention_v1 : forward behaviour
# ---------------------------------------------------------------------------
def test_v1_output_shape():
    out = make_v1()(INPUTS)
    assert out.shape == (T, D_OUT)


def test_v1_output_is_finite():
    out = make_v1()(INPUTS)
    assert torch.isfinite(out).all()


def test_v1_matches_manual_scaled_dot_product_attention():
    m = make_v1()
    expected, _ = manual_attention(INPUTS, m.W_query, m.W_key, m.W_value)
    assert torch.allclose(m(INPUTS), expected, atol = 1e-6)


def test_v1_same_seed_gives_same_output():
    assert torch.allclose(make_v1(seed = 1)(INPUTS), make_v1(seed = 1)(INPUTS))


def test_v1_different_seed_gives_different_output():
    assert not torch.allclose(make_v1(seed = 1)(INPUTS), make_v1(seed = 2)(INPUTS))


# ---------------------------------------------------------------------------
# SelfAttention_v2 : structure
# ---------------------------------------------------------------------------
def test_v2_uses_linear_layers():
    m = make_v2()

    for layer in (m.W_query, m.W_key, m.W_value):
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == D_IN
        assert layer.out_features == D_OUT


def test_v2_no_bias_by_default():
    m = make_v2()
    assert m.W_query.bias is None
    assert m.W_key.bias is None
    assert m.W_value.bias is None
    assert len(list(m.parameters())) == 3


def test_v2_qkv_bias_adds_bias_parameters():
    m = make_v2(qkv_bias = True)
    assert m.W_query.bias.shape == (D_OUT,)
    assert len(list(m.parameters())) == 6


# ---------------------------------------------------------------------------
# SelfAttention_v2 : forward behaviour
# ---------------------------------------------------------------------------
def test_v2_output_shape():
    out = make_v2()(INPUTS)
    assert out.shape == (T, D_OUT)


def test_v2_output_is_finite():
    assert torch.isfinite(make_v2()(INPUTS)).all()


def test_v2_matches_manual_scaled_dot_product_attention():
    m = make_v2()
    expected, _ = manual_attention(INPUTS, m.W_query.weight.T, m.W_key.weight.T, m.W_value.weight.T)
    assert torch.allclose(m(INPUTS), expected, atol = 1e-6)


def test_v2_bias_changes_output():
    torch.manual_seed(0)
    m = SelfAttention_v2(D_IN, D_OUT, qkv_bias = True)
    before = m(INPUTS).detach().clone()

    with torch.no_grad():
        m.W_value.bias.add_(1.0)     # shift every value vector
    
    assert not torch.allclose(before, m(INPUTS))


# ---------------------------------------------------------------------------
# v1 <-> v2 equivalence (book exercise 3.1)
# ---------------------------------------------------------------------------
def test_v1_equals_v2_when_weights_are_shared():
    v1, v2 = make_v1(), make_v2()
    copy_v2_into_v1(v1, v2)
    assert torch.allclose(v1(INPUTS), v2(INPUTS), atol = 1e-6)


def test_v1_and_v2_differ_with_independent_init():
    assert not torch.allclose(make_v1()(INPUTS), make_v2()(INPUTS))


# ---------------------------------------------------------------------------
# Mathematical properties (apply to both versions)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [make_v1, make_v2])
def test_attention_weights_rows_sum_to_one_and_are_positive(factory):
    m = factory()
    
    if isinstance(m, SelfAttention_v1):
        W_q, W_k, W_v = m.W_query, m.W_key, m.W_value
    else:
        W_q, W_k, W_v = m.W_query.weight.T, m.W_key.weight.T, m.W_value.weight.T
    
    _, weights = manual_attention(INPUTS, W_q, W_k, W_v)
    assert weights.shape == (T, T)
    assert torch.allclose(weights.sum(dim = -1), torch.ones(T), atol = 1e-6)
    assert (weights > 0).all()


@pytest.mark.parametrize("factory", [make_v1, make_v2])
def test_output_is_convex_combination_of_values(factory):
    """
        Each output row = weighted avg of value vectors -> must lie within their min/max.
    """
    m = factory()
    if isinstance(m, SelfAttention_v1):
        values = INPUTS @ m.W_value
    else:
        values = m.W_value(INPUTS)
    out = m(INPUTS)
    assert (out <= values.max(dim = 0).values + 1e-6).all()
    assert (out >= values.min(dim = 0).values - 1e-6).all()


@pytest.mark.parametrize("factory", [make_v1, make_v2])
def test_permuting_tokens_permutes_output_rows(factory):
    """
        No positional info and no mask -> attention is permutation-equivariant.
    """
    m = factory()
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    assert torch.allclose(m(INPUTS[perm]), m(INPUTS)[perm], atol = 1e-6)


@pytest.mark.parametrize("factory", [make_v1, make_v2])
def test_single_token_returns_its_own_value_vector(factory):
    """
        With one token, softmax over one score = 1 -> output equals that token's value.
    """
    m = factory()
    x = INPUTS[:1]

    if isinstance(m, SelfAttention_v1):
        value = x @ m.W_value
    else:
        value = m.W_value(x)
    
    assert torch.allclose(m(x), value, atol = 1e-6)


@pytest.mark.parametrize("factory", [make_v1, make_v2])
def test_gradients_flow_to_all_parameters(factory):
    m = factory()
    m(INPUTS).sum().backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0, f"zero gradient for {name}"


@pytest.mark.parametrize("factory", [make_v1, make_v2])
def test_scaling_by_sqrt_dk_is_applied(factory):
    """
        Module output must match the sqrt(d_k)-scaled softmax and NOT the unscaled one.
    """
    m = factory()

    if isinstance(m, SelfAttention_v1):
        W_q, W_k, W_v = m.W_query, m.W_key, m.W_value
    else:
        W_q, W_k, W_v = m.W_query.weight.T, m.W_key.weight.T, m.W_value.weight.T
    
    scores = (INPUTS @ W_q) @ (INPUTS @ W_k).T
    values = INPUTS @ W_v
    scaled = torch.softmax(scores / D_OUT ** 0.5, dim = -1) @ values
    unscaled = torch.softmax(scores, dim = -1) @ values
    out = m(INPUTS)
    assert torch.allclose(out, scaled, atol = 1e-6)
    assert not torch.allclose(out, unscaled, atol = 1e-4)
