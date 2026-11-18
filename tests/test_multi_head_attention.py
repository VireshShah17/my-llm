import pytest
import torch
import torch.nn as nn
from src.attention.attention import INPUTS
from src.attention.causal_attention import CausalAttention
from src.attention.multi_head_attention import (
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
)

D_IN = 3
T = 6


def make_batch(batch_size = 2):
    return torch.stack([INPUTS] * batch_size, dim = 0)      # [b, 6, 3]


def make_wrapper(d_out = 2, num_heads = 2, dropout = 0.0, context_length = T,
                 qkv_bias = False, seed = 123):
    torch.manual_seed(seed)
    return MultiHeadAttentionWrapper(
        D_IN, d_out, context_length, dropout, num_heads = num_heads, qkv_bias = qkv_bias
    )


def make_mha(d_out = 4, num_heads = 2, dropout = 0.0, context_length = T,
             qkv_bias = False, seed = 123):
    torch.manual_seed(seed)
    return MultiHeadAttention(
        D_IN, d_out, context_length, dropout, num_heads = num_heads, qkv_bias = qkv_bias
    )


def set_identity_out_proj(m):
    """
        Make out_proj a no-op so the module output is exactly the concatenated head outputs.
    """
    with torch.no_grad():
        m.out_proj.weight.copy_(torch.eye(m.d_out))
        m.out_proj.bias.zero_()


def single_head_context(x, W_q, W_k, W_v, head_dim):
    """
        Reference causal attention for ONE head.
        W_* are nn.Linear-style weights of shape [head_dim, d_in].
    """
    queries, keys, values = x @ W_q.T, x @ W_k.T, x @ W_v.T
    scores = queries @ keys.transpose(1, 2)
    n = x.shape[1]
    future = torch.triu(torch.ones(n, n), diagonal = 1).bool()
    weights = torch.softmax(scores.masked_fill(future, -torch.inf) / head_dim ** 0.5, dim = -1)

    return weights @ values


def head_slices(mha, h):
    """
        Rows [h*head_dim : (h+1)*head_dim] of the big Q/K/V matrices belong to head h.
    """
    hd = mha.head_dim
    sl = slice(h * hd, (h + 1) * hd)

    return mha.W_query.weight[sl], mha.W_key.weight[sl], mha.W_value.weight[sl]


# ===========================================================================
# Part A : MultiHeadAttentionWrapper
# ===========================================================================
def test_wrapper_holds_module_list_of_causal_attention_heads():
    m = make_wrapper(num_heads = 3)
    assert isinstance(m.heads, nn.ModuleList)
    assert len(m.heads) == 3
    assert all(isinstance(h, CausalAttention) for h in m.heads)


def test_wrapper_output_shape_d_out_2_heads_2():
    assert make_wrapper(d_out = 2, num_heads = 2)(make_batch()).shape == (2, T, 4)


@pytest.mark.parametrize("d_out, num_heads", [(2, 1), (2, 2), (2, 3), (4, 2)])
def test_wrapper_output_dim_is_d_out_times_num_heads(d_out, num_heads):
    out = make_wrapper(d_out = d_out, num_heads = num_heads)(make_batch())
    assert out.shape == (2, T, d_out * num_heads)


def test_wrapper_output_is_concatenation_of_head_outputs():
    m = make_wrapper(d_out = 2, num_heads = 3)
    x = make_batch()
    expected = torch.cat([head(x) for head in m.heads], dim = -1)
    assert torch.allclose(m(x), expected)


def test_wrapper_heads_have_independent_weights():
    m = make_wrapper(num_heads = 2)
    assert not torch.allclose(m.heads[0].W_query.weight, m.heads[1].W_query.weight) # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
    assert not torch.allclose(m.heads[0].W_value.weight, m.heads[1].W_value.weight) # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]


def test_wrapper_passes_config_to_every_head():
    m = make_wrapper(num_heads = 2, dropout = 0.3, context_length = 8, qkv_bias = True)

    for head in m.heads:
        assert head.dropout.p == 0.3 # type: ignore
        assert head.mask.shape == (8, 8)
        assert head.W_query.bias is not None # type: ignore


def test_wrapper_parameter_count():
    heads, d_out = 3, 2
    m = make_wrapper(d_out = d_out, num_heads = heads)
    assert sum(p.numel() for p in m.parameters()) == heads * 3 * D_IN * d_out


def test_wrapper_is_causal():
    m = make_wrapper()
    x = make_batch()
    changed = x.clone()
    changed[:, -1] += 1.0
    assert torch.allclose(m(changed)[:, :-1], m(x)[:, :-1], atol = 1e-6)


def test_wrapper_gradients_flow_to_all_parameters():
    m = make_wrapper(qkv_bias = True)
    m(make_batch()).sum().backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"


# ===========================================================================
# Part B : MultiHeadAttention (efficient weight-split version)
# ===========================================================================
# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_mha_output_shape_d_out_2_heads_2():
    m = MultiHeadAttention(3, 2, 6, 0.0, num_heads = 2)
    assert m(make_batch()).shape == (2, 6, 2)


@pytest.mark.parametrize("d_out, num_heads", [(2, 1), (4, 2), (4, 4), (6, 3), (8, 2)])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_mha_output_shape_is_batch_tokens_d_out(d_out, num_heads, batch_size):
    out = make_mha(d_out = d_out, num_heads = num_heads)(make_batch(batch_size))
    assert out.shape == (batch_size, T, d_out)


def test_mha_requires_d_out_divisible_by_num_heads():
    with pytest.raises(AssertionError):
        MultiHeadAttention(3, 3, 6, 0.0, num_heads = 2)


def test_mha_head_dim_is_d_out_over_num_heads():
    m = make_mha(d_out = 6, num_heads = 3)
    assert m.d_out == 6
    assert m.num_heads == 3
    assert m.head_dim == 2


def test_mha_uses_linear_layers_with_correct_sizes():
    m = make_mha(d_out = 4, num_heads = 2)

    for layer in (m.W_query, m.W_key, m.W_value):
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == D_IN
        assert layer.out_features == 4
        assert layer.bias is None
                           # qkv_bias defaults to False
    assert isinstance(m.out_proj, nn.Linear)
    assert (m.out_proj.in_features, m.out_proj.out_features) == (4, 4)
    assert m.out_proj.bias is not None                  # out_proj always has a bias


def test_mha_qkv_bias_flag_adds_bias_to_q_k_v_only():
    m = make_mha(qkv_bias = True)
    assert m.W_query.bias is not None
    assert m.W_key.bias is not None
    assert m.W_value.bias is not None


def test_mha_mask_is_strictly_upper_triangular_buffer():
    m = make_mha(context_length = 8)
    assert m.mask.shape == (8, 8)
    assert torch.equal(m.mask.bool(), torch.triu(torch.ones(8, 8), diagonal = 1).bool())
    assert "mask" in dict(m.named_buffers())
    assert "mask" not in dict(m.named_parameters())
    assert "mask" in m.state_dict()


def test_mha_state_dict_roundtrip_preserves_output():
    src, dst = make_mha(seed = 1), make_mha(seed = 2)
    dst.load_state_dict(src.state_dict())
    x = make_batch()
    assert torch.allclose(src(x), dst(x))


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d_out, num_heads", [(2, 2), (4, 2), (4, 4), (6, 3)])
def test_mha_parameter_count_without_qkv_bias(d_out, num_heads):
    m = make_mha(d_out = d_out, num_heads = num_heads)
    expected = 3 * D_IN * d_out + d_out * d_out + d_out
    assert sum(p.numel() for p in m.parameters()) == expected


def test_mha_parameter_count_with_qkv_bias():
    d_out = 4
    m = make_mha(d_out = d_out, num_heads = 2, qkv_bias = True)
    expected = 3 * (D_IN * d_out + d_out) + d_out * d_out + d_out
    assert sum(p.numel() for p in m.parameters()) == expected


def test_mha_parameter_count_does_not_depend_on_num_heads():
    """
        Splitting into heads re-uses the same big matrices -> same size for 1, 2 or 4 heads.
    """
    counts = {h: sum(p.numel() for p in make_mha(d_out = 4, num_heads = h).parameters())
              for h in (1, 2, 4)}
    assert len(set(counts.values())) == 1


# ---------------------------------------------------------------------------
# Causality (catches a wrong mask after the reshape / transpose)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_mha_changing_last_token_does_not_change_earlier_outputs(num_heads):
    m = make_mha(d_out = 4, num_heads = num_heads)
    x = make_batch()
    changed = x.clone()
    changed[:, -1] += 1.0
    assert torch.allclose(m(changed)[:, :-1], m(x)[:, :-1], atol = 1e-6)


@pytest.mark.parametrize("k", [0, 2, 4])
def test_mha_changing_token_k_leaves_earlier_outputs_untouched_but_affects_k(k):
    m = make_mha()
    x = make_batch()
    changed = x.clone()
    changed[:, k] += 1.0
    out_old, out_new = m(x), m(changed)

    if k > 0:
        assert torch.allclose(out_new[:, :k], out_old[:, :k], atol = 1e-6)
    assert not torch.allclose(out_new[:, k], out_old[:, k])


def test_mha_changing_an_early_token_does_affect_later_outputs():
    m = make_mha()
    x = make_batch()
    changed = x.clone()
    changed[:, 0] += 1.0
    assert not torch.allclose(m(changed)[:, -1], m(x)[:, -1])


def test_mha_gradient_of_first_output_ignores_later_tokens():
    m = make_mha()
    x = make_batch().clone().requires_grad_(True)
    m(x)[:, 0].sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad[:, 1:], torch.zeros_like(x.grad[:, 1:]))
    assert x.grad[:, 0].abs().sum() > 0


def test_mha_shorter_sequence_equals_prefix_of_full_output():
    m = make_mha()
    x = make_batch()
    assert torch.allclose(m(x[:, :3]), m(x)[:, :3], atol = 1e-6)


def test_mha_shorter_than_context_length_works():
    m = make_mha(context_length = 10)
    assert m(make_batch()[:, :4]).shape == (2, 4, 4)


def test_mha_sequence_longer_than_context_length_raises():
    m = make_mha(context_length = 4)
    with pytest.raises(RuntimeError):
        m(make_batch())


# ---------------------------------------------------------------------------
# Head split equivalence : the key test for view / transpose correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d_out, num_heads", [(2, 1), (4, 2), (4, 4), (6, 3)])
def test_mha_per_head_outputs_match_single_head_computations(d_out, num_heads):
    """
        With out_proj = identity, output[..., h*hd:(h+1)*hd] must equal a plain
        single-head computation that uses rows h*hd:(h+1)*hd of the Q/K/V weights.
    """
    m = make_mha(d_out = d_out, num_heads = num_heads)
    set_identity_out_proj(m)
    x = make_batch()
    out = m(x)
    hd = m.head_dim

    for h in range(num_heads):
        W_q, W_k, W_v = head_slices(m, h)
        expected_h = single_head_context(x, W_q, W_k, W_v, hd)
        assert torch.allclose(out[..., h * hd:(h + 1) * hd], expected_h, atol = 1e-6), \
            f"head {h} does not match its single-head reference"


def test_mha_out_proj_is_applied_to_concatenated_heads():
    m = make_mha(d_out = 4, num_heads = 2)
    x = make_batch()
    hd = m.head_dim
    heads = torch.cat(
        [single_head_context(x, *head_slices(m, h), hd) for h in range(2)], dim = -1
    )
    expected = heads @ m.out_proj.weight.T + m.out_proj.bias
    assert torch.allclose(m(x), expected, atol = 1e-6)


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_mha_equals_wrapper_when_weights_are_copied_and_out_proj_is_identity(num_heads):
    """
        The efficient version is the wrapper with one big matrix instead of many small ones.
    """
    d_out = 4
    mha = make_mha(d_out = d_out, num_heads = num_heads)
    set_identity_out_proj(mha)
    hd = mha.head_dim

    wrapper = make_wrapper(d_out = hd, num_heads = num_heads)

    with torch.no_grad():
        for h, head in enumerate(wrapper.heads):
            W_q, W_k, W_v = head_slices(mha, h)
            head.W_query.weight.copy_(W_q) # type: ignore
            head.W_key.weight.copy_(W_k) # type: ignore
            head.W_value.weight.copy_(W_v) # type: ignore

    x = make_batch()
    assert wrapper(x).shape == mha(x).shape == (2, T, d_out)
    assert torch.allclose(wrapper(x), mha(x), atol = 1e-6)


def test_mha_heads_are_independent():
    """
        Changing the value weights of head 1 must not change head 0's slice of the output.
    """
    m = make_mha(d_out = 4, num_heads = 2)
    set_identity_out_proj(m)
    x = make_batch()
    before = m(x).detach().clone()
    hd = m.head_dim

    with torch.no_grad():
        m.W_value.weight[hd:] += 1.0                    # head 1 only

    after = m(x)
    assert torch.allclose(after[..., :hd], before[..., :hd], atol = 1e-6)
    assert not torch.allclose(after[..., hd:], before[..., hd:])


def test_mha_heads_learn_different_patterns_with_different_weights():
    """
        Concept check: two heads with different weights give different outputs.
    """
    m = make_mha(d_out = 4, num_heads = 2)
    set_identity_out_proj(m)
    out = m(make_batch())
    hd = m.head_dim
    assert not torch.allclose(out[..., :hd], out[..., hd:])


# ---------------------------------------------------------------------------
# Scaling : sqrt(head_dim), NOT sqrt(d_out)
# ---------------------------------------------------------------------------
def test_mha_scaling_uses_sqrt_head_dim_not_sqrt_d_out():
    d_out, num_heads = 4, 2                              # head_dim = 2, so sqrt(2) vs sqrt(4)
    m = make_mha(d_out = d_out, num_heads = num_heads)
    set_identity_out_proj(m)
    with torch.no_grad():                                # sharpen scores so scale choice matters
        m.W_query.weight.mul_(4)
        m.W_key.weight.mul_(4)
    x = make_batch()
    hd = m.head_dim

    def reference(scale_dim):
        outs = []
        for h in range(num_heads):
            W_q, W_k, W_v = head_slices(m, h)
            q, k, v = x @ W_q.T, x @ W_k.T, x @ W_v.T
            scores = (q @ k.transpose(1, 2)).masked_fill(
                torch.triu(torch.ones(T, T), diagonal = 1).bool(), -torch.inf
            )
            outs.append(torch.softmax(scores / scale_dim ** 0.5, dim = -1) @ v)
        return torch.cat(outs, dim = -1)

    out = m(x)
    assert torch.allclose(out, reference(hd), atol = 1e-6)
    assert not torch.allclose(out, reference(d_out), atol = 1e-4)


# ---------------------------------------------------------------------------
# Other behaviour
# ---------------------------------------------------------------------------
def test_mha_output_is_finite():
    assert torch.isfinite(make_mha()(make_batch())).all()


def test_mha_batch_items_are_processed_independently():
    m = make_mha()
    a = INPUTS.unsqueeze(0)
    b = (INPUTS.flip(0) * 0.5).unsqueeze(0)
    together = m(torch.cat([a, b], dim = 0))
    assert torch.allclose(together[0], m(a)[0], atol = 1e-6)
    assert torch.allclose(together[1], m(b)[0], atol = 1e-6)


def test_mha_forward_does_not_modify_the_input():
    m = make_mha()
    x = make_batch()
    before = x.clone()
    m(x)
    assert torch.equal(x, before)


def test_mha_gradients_flow_to_all_parameters_including_out_proj():
    m = make_mha(qkv_bias = True)
    m(make_batch()).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all()
    assert m.out_proj.weight.grad.abs().sum() > 0 # type: ignore


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------
def test_mha_dropout_module_uses_given_probability():
    assert make_mha(dropout = 0.3).dropout.p == 0.3


def test_mha_eval_mode_ignores_dropout():
    x = make_batch()
    assert torch.allclose(
        make_mha(dropout = 0.5).eval()(x), make_mha(dropout = 0.0).eval()(x), atol = 1e-6
    )


def test_mha_train_mode_dropout_is_random():
    m = make_mha(dropout = 0.5).train()
    x = make_batch()
    torch.manual_seed(0)
    first = m(x)
    torch.manual_seed(1)
    second = m(x)
    assert not torch.allclose(first, second)


def test_mha_train_mode_with_zero_dropout_is_deterministic():
    m = make_mha(dropout = 0.0).train()
    x = make_batch()
    assert torch.allclose(m(x), m(x))


# ---------------------------------------------------------------------------
# Exercise 3.3 : GPT-2 (small) scale. Deliberately NO big forward pass (4 GB RAM).
# ---------------------------------------------------------------------------
def test_gpt2_small_attention_parameter_count_and_tiny_forward_shape():
    m = MultiHeadAttention(768, 768, 1024, 0.0, num_heads = 12)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"\nGPT-2 small MultiHeadAttention parameters: {n_params:,}")

    assert m.head_dim == 64
    assert n_params == 3 * 768 ** 2 + 768 ** 2 + 768 == 2_360_064

    with torch.no_grad():
        out = m(torch.rand(1, 8, 768))                   # tiny input only
    assert out.shape == (1, 8, 768)
