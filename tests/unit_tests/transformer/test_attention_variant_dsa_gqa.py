import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
import pytest

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.transformer.experimental_attention_variant.dsa import (
    fused_qk_topk_chunked,
    fused_qk_topk_naive,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGroupedSelfAttention,
    _build_shifted_causal_mask,
    compute_gqa_dsa_indexer_loss,
    unfused_grouped_dsa_fn,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    DSAMinMemoryGQAFn,
    _forward_min_memory_impl,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _DummyTPGroup:
    def size(self):
        return 1


class _DummyPGCollection:
    tp = _DummyTPGroup()


class _DummyRotary:
    def __init__(self, rotary_dim: int, rotary_interleaved: bool = False):
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.rotary_interleaved = rotary_interleaved
        self.seq_len_interpolation_factor = None


def test_mamba_stack_spec_uses_dsa_grouped_self_attention():
    attention_module = mamba_stack_spec.submodules.attention_layer.submodules.self_attention.module
    assert attention_module is DSGroupedSelfAttention


def _causal_mask(seqlen: int, device: torch.device):
    return torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), dtype=torch.float32, device=device),
        diagonal=1,
    )


def _causal_index_scores(index_scores: torch.Tensor):
    masked_scores = index_scores + _causal_mask(
        index_scores.size(1), index_scores.device
    ).view(1, index_scores.size(1), index_scores.size(2))
    return masked_scores.detach().requires_grad_(index_scores.requires_grad)


def _random_topk_indices(batch_size: int, seqlen: int, topk: int):
    return torch.randn(batch_size, seqlen, seqlen).topk(topk, dim=-1).indices


def _rotary_freqs(rotary, seqlen: int, rotary_dim: int):
    positions = torch.arange(seqlen, dtype=rotary.inv_freq.dtype, device=rotary.inv_freq.device)
    freqs = torch.outer(positions, rotary.inv_freq[: rotary_dim // 2])
    if not rotary.rotary_interleaved:
        freqs = torch.cat((freqs, freqs), dim=-1)
    else:
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=-2)
    return freqs[:, None, None, :]


def _apply_reference_indexer_rope(
    x: torch.Tensor, rotary, config_rotary_interleaved: bool, rotary_dim: int
):
    x_nope, x_pe = torch.split(x, [x.size(-1) - rotary_dim, rotary_dim], dim=-1)
    x_pe = _apply_rotary_pos_emb_bshd(
        x_pe,
        _rotary_freqs(rotary, x.size(0), rotary_dim),
        rotary_interleaved=config_rotary_interleaved,
        multi_latent_attention=False,
        mscale=1.0,
    )
    return torch.cat([x_nope, x_pe], dim=-1)


def test_transformer_config_accepts_min_memory_backend():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_kernel_backend="triton-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_sparse_loss_use_topk_only=True,
        dsa_sparse_attention_query_chunk_size=2,
        dsa_indexer_use_hadamard=True,
    )

    assert config.dsa_kernel_backend == "triton-min-memory"


def test_transformer_config_min_memory_accepts_sparse_loss_without_topk_only_flag():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_kernel_backend="triton-min-memory",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_use_hadamard=True,
    )

    assert config.dsa_indexer_use_sparse_loss
    assert not config.dsa_indexer_sparse_loss_use_topk_only


def test_transformer_config_min_memory_requires_sparse_loss():
    with pytest.raises(AssertionError, match="dsa_indexer_use_sparse_loss"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=4,
            dsa_kernel_backend="triton-min-memory",
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_hadamard=True,
        )


def test_min_memory_impl_matches_reference_forward_and_loss():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 6
    hidden_size = 16
    num_heads = 4
    num_query_groups = 2
    head_dim = 8
    index_heads = 2
    index_head_dim = 4
    topk = 3
    loss_coeff = 0.7

    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    query = torch.randn(seqlen, batch_size, num_heads, head_dim)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    linear_q_weight = torch.randn(index_heads * index_head_dim, hidden_size)
    linear_k_weight = torch.randn(index_head_dim, hidden_size)
    k_norm_weight = torch.randn(index_head_dim)
    k_norm_bias = torch.randn(index_head_dim)
    linear_weights_weight = torch.randn(index_heads, hidden_size)
    pg_collection = _DummyPGCollection()

    q_index = F.linear(hidden_states, linear_q_weight).reshape(
        seqlen, batch_size, index_heads, index_head_dim
    )
    k_index = F.layer_norm(
        F.linear(hidden_states, linear_k_weight),
        (index_head_dim,),
        k_norm_weight,
        k_norm_bias,
    )
    weights = F.linear(hidden_states, linear_weights_weight)
    weights = weights * (index_heads**-0.5) * (index_head_dim**-0.5)
    index_scores, topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, topk, _causal_mask(seqlen, hidden_states.device)
    )
    reference_output = unfused_grouped_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        head_dim**-0.5,
        use_gather=True,
    )
    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=loss_coeff,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=index_scores.gather(-1, topk_indices),
    )

    output, loss = _forward_min_memory_impl(
        query,
        key,
        value,
        hidden_states,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        True,
        linear_weights_weight,
        1e-5,
        index_heads,
        index_head_dim,
        topk,
        0,
        None,
        False,
        False,
        head_dim**-0.5,
        loss_coeff,
        2,
        3,
        pg_collection,
        compute_loss=True,
    )

    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(loss, reference_loss)


def test_min_memory_impl_matches_reference_rope_interleaved_layout():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 6
    hidden_size = 14
    num_heads = 4
    num_query_groups = 2
    head_dim = 4
    index_heads = 2
    index_head_dim = 6
    rotary_dim = 4
    topk = 3
    loss_coeff = 0.7
    config_rotary_interleaved = True

    hidden_states = torch.randn(seqlen, batch_size, hidden_size)
    query = torch.randn(seqlen, batch_size, num_heads, head_dim)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim)
    linear_q_weight = torch.randn(index_heads * index_head_dim, hidden_size)
    linear_k_weight = torch.randn(index_head_dim, hidden_size)
    k_norm_weight = torch.randn(index_head_dim)
    k_norm_bias = torch.randn(index_head_dim)
    linear_weights_weight = torch.randn(index_heads, hidden_size)
    pg_collection = _DummyPGCollection()
    rotary = _DummyRotary(rotary_dim, rotary_interleaved=False)

    q_index = F.linear(hidden_states, linear_q_weight).reshape(
        seqlen, batch_size, index_heads, index_head_dim
    )
    q_index = _apply_reference_indexer_rope(
        q_index, rotary, config_rotary_interleaved, rotary_dim
    )
    k_index = F.layer_norm(
        F.linear(hidden_states, linear_k_weight),
        (index_head_dim,),
        k_norm_weight,
        k_norm_bias,
    ).reshape(seqlen, batch_size, 1, index_head_dim)
    k_index = _apply_reference_indexer_rope(
        k_index, rotary, config_rotary_interleaved, rotary_dim
    ).reshape(seqlen, batch_size, index_head_dim)
    weights = F.linear(hidden_states, linear_weights_weight)
    weights = weights * (index_heads**-0.5) * (index_head_dim**-0.5)

    index_scores, topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, topk, _causal_mask(seqlen, hidden_states.device)
    )
    reference_output = unfused_grouped_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        head_dim**-0.5,
        use_gather=True,
    )
    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=loss_coeff,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=index_scores.gather(-1, topk_indices),
    )

    output, loss = _forward_min_memory_impl(
        query,
        key,
        value,
        hidden_states,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        True,
        linear_weights_weight,
        1e-5,
        index_heads,
        index_head_dim,
        topk,
        rotary_dim,
        rotary,
        True,
        False,
        head_dim**-0.5,
        loss_coeff,
        2,
        3,
        pg_collection,
        compute_loss=True,
        rotary_interleaved=config_rotary_interleaved,
    )

    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(loss, reference_loss)


def test_min_memory_impl_matches_reference_gradients():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 5
    hidden_size = 12
    num_heads = 4
    num_query_groups = 2
    head_dim = 4
    index_heads = 2
    index_head_dim = 4
    topk = 3
    loss_coeff = 0.7
    pg_collection = _DummyPGCollection()

    def _make_tensors():
        hidden_states = torch.randn(seqlen, batch_size, hidden_size)
        query = torch.randn(seqlen, batch_size, num_heads, head_dim, requires_grad=True)
        key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, requires_grad=True)
        value = torch.randn(seqlen, batch_size, num_query_groups, head_dim, requires_grad=True)
        linear_q_weight = torch.randn(
            index_heads * index_head_dim, hidden_size, requires_grad=True
        )
        linear_k_weight = torch.randn(index_head_dim, hidden_size, requires_grad=True)
        k_norm_weight = torch.randn(index_head_dim, requires_grad=True)
        k_norm_bias = torch.randn(index_head_dim, requires_grad=True)
        linear_weights_weight = torch.randn(index_heads, hidden_size, requires_grad=True)
        return (
            hidden_states,
            query,
            key,
            value,
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        )

    min_tensors = _make_tensors()
    ref_tensors = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in min_tensors)

    (
        hidden_states,
        query,
        key,
        value,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        linear_weights_weight,
    ) = min_tensors
    output, loss = DSAMinMemoryGQAFn.apply(
        query,
        key,
        value,
        hidden_states,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        linear_weights_weight,
        True,
        1e-5,
        index_heads,
        index_head_dim,
        topk,
        0,
        None,
        False,
        False,
        head_dim**-0.5,
        loss_coeff,
        2,
        3,
        pg_collection,
    )
    (output.sum() + loss).backward()

    (
        ref_hidden_states,
        ref_query,
        ref_key,
        ref_value,
        ref_linear_q_weight,
        ref_linear_k_weight,
        ref_k_norm_weight,
        ref_k_norm_bias,
        ref_linear_weights_weight,
    ) = ref_tensors
    q_index = F.linear(ref_hidden_states, ref_linear_q_weight).reshape(
        seqlen, batch_size, index_heads, index_head_dim
    )
    k_index = F.layer_norm(
        F.linear(ref_hidden_states, ref_linear_k_weight),
        (index_head_dim,),
        ref_k_norm_weight,
        ref_k_norm_bias,
    )
    weights = F.linear(ref_hidden_states, ref_linear_weights_weight)
    weights = weights * (index_heads**-0.5) * (index_head_dim**-0.5)
    index_scores, topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, topk, _causal_mask(seqlen, ref_hidden_states.device)
    )
    ref_output = unfused_grouped_dsa_fn(
        ref_query,
        ref_key,
        ref_value,
        topk_indices,
        head_dim**-0.5,
        use_gather=True,
    )
    ref_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=ref_query.detach(),
        key=ref_key.detach(),
        softmax_scale=head_dim**-0.5,
        loss_coeff=loss_coeff,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=index_scores.gather(-1, topk_indices),
    )
    (ref_output.sum() + ref_loss).backward()

    for min_tensor, ref_tensor in zip(min_tensors[1:], ref_tensors[1:]):
        torch.testing.assert_close(min_tensor.grad, ref_tensor.grad)


def test_compute_gqa_dsa_indexer_loss_dense_and_sparse():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32)
    )
    topk_indices = index_scores.topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    dense_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores.clone(),
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=False,
        pg_collection=pg_collection,
    )
    sparse_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores.clone(),
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
    )

    assert dense_loss.ndim == 0
    assert sparse_loss.ndim == 0
    assert torch.isfinite(dense_loss)
    assert torch.isfinite(sparse_loss)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_matches_reference():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
    )
    reference_loss.backward()
    reference_grad = index_scores.grad.clone()

    index_scores.grad = None

    sparse_topk_only_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
    )
    sparse_topk_only_loss.backward()

    torch.testing.assert_close(sparse_topk_only_loss, reference_loss)
    torch.testing.assert_close(index_scores.grad, reference_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_chunked_matches_unchunked():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    unchunked_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
    )
    unchunked_loss.backward()
    unchunked_grad = index_scores.grad.clone()

    index_scores.grad = None

    chunked_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        query_chunk_size=3,
    )
    chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, unchunked_loss)
    torch.testing.assert_close(index_scores.grad, unchunked_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_selected_scores_matches_reference():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    selected_index_scores = (
        index_scores.detach().gather(-1, topk_indices).clone().requires_grad_(True)
    )
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
    )
    reference_loss.backward()
    reference_grad = index_scores.grad.gather(-1, topk_indices)

    selected_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        selected_index_scores=selected_index_scores,
    )
    selected_loss.backward()

    torch.testing.assert_close(selected_loss, reference_loss)
    torch.testing.assert_close(selected_index_scores.grad, reference_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_selected_scores_chunked_matches_reference():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    selected_index_scores = (
        index_scores.detach().gather(-1, topk_indices).clone().requires_grad_(True)
    )
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    reference_loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        query_chunk_size=3,
    )
    reference_loss.backward()
    reference_grad = index_scores.grad.gather(-1, topk_indices)

    selected_loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=query,
        key=key,
        softmax_scale=head_dim**-0.5,
        loss_coeff=0.7,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        query_chunk_size=3,
        selected_index_scores=selected_index_scores,
    )
    selected_loss.backward()

    torch.testing.assert_close(selected_loss, reference_loss)
    torch.testing.assert_close(selected_index_scores.grad, reference_grad)


def test_unfused_grouped_dsa_fn_output_shape():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    value = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)

    output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
    )

    assert output.shape == (seqlen, batch_size, num_heads * head_dim)
    assert output.dtype == query.dtype


def test_unfused_grouped_dsa_fn_matches_dense_reference():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
    mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    sparse_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
        use_gather=True,
    )
    sparse_output.sum().backward()
    sparse_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    dense_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
    )
    dense_output.sum().backward()

    torch.testing.assert_close(sparse_output, dense_output)
    torch.testing.assert_close(query.grad, sparse_grads[0])
    torch.testing.assert_close(key.grad, sparse_grads[1])
    torch.testing.assert_close(value.grad, sparse_grads[2])


def test_unfused_grouped_dsa_fn_gather_bool_mask_matches_dense_float_mask():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
    bool_mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.bool)
    bool_mask[:, :, -1] = True
    float_mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.float32).masked_fill(
        bool_mask, float("-inf")
    )

    gather_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=bool_mask,
        use_gather=True,
    )
    gather_output.sum().backward()
    gather_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    dense_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=float_mask,
    )
    dense_output.sum().backward()

    torch.testing.assert_close(gather_output, dense_output)
    torch.testing.assert_close(query.grad, gather_grads[0])
    torch.testing.assert_close(key.grad, gather_grads[1])
    torch.testing.assert_close(value.grad, gather_grads[2])


def test_unfused_grouped_dsa_fn_chunked_matches_unchunked():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
    mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    unchunked_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
        use_gather=True,
    )
    unchunked_output.sum().backward()
    unchunked_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    chunked_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
        query_chunk_size=2,
        use_gather=True,
    )
    chunked_output.sum().backward()

    torch.testing.assert_close(chunked_output, unchunked_output)
    torch.testing.assert_close(query.grad, unchunked_grads[0])
    torch.testing.assert_close(key.grad, unchunked_grads[1])
    torch.testing.assert_close(value.grad, unchunked_grads[2])


def test_unfused_grouped_dsa_fn_recompute_matches_normal():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
    mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    normal_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
    )
    normal_output.sum().backward()
    normal_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    def _compute_recompute_output(
        query_tensor: torch.Tensor, key_tensor: torch.Tensor, value_tensor: torch.Tensor
    ) -> torch.Tensor:
        return unfused_grouped_dsa_fn(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            topk_indices=topk_indices,
            softmax_scale=head_dim**-0.5,
            mask=mask,
        )

    recompute_output = torch_checkpoint.checkpoint(
        _compute_recompute_output,
        query,
        key,
        value,
        use_reentrant=False,
    )
    recompute_output.sum().backward()

    torch.testing.assert_close(recompute_output, normal_output)
    torch.testing.assert_close(query.grad, normal_grads[0])
    torch.testing.assert_close(key.grad, normal_grads[1])
    torch.testing.assert_close(value.grad, normal_grads[2])


def test_unfused_grouped_dsa_fn_gather_recompute_matches_normal():
    torch.manual_seed(123)

    seqlen = 6
    batch_size = 2
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 3

    query = torch.randn(
        seqlen, batch_size, num_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    key = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    value = torch.randn(
        seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32, requires_grad=True
    )
    topk_indices = _random_topk_indices(batch_size, seqlen, topk)
    mask = torch.zeros(batch_size, seqlen, seqlen, dtype=torch.bool)
    mask[:, :, -1] = True

    normal_output = unfused_grouped_dsa_fn(
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        softmax_scale=head_dim**-0.5,
        mask=mask,
        query_chunk_size=2,
        use_gather=True,
    )
    normal_output.sum().backward()
    normal_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())

    query.grad = None
    key.grad = None
    value.grad = None

    def _compute_recompute_output(
        query_tensor: torch.Tensor, key_tensor: torch.Tensor, value_tensor: torch.Tensor
    ) -> torch.Tensor:
        return unfused_grouped_dsa_fn(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            topk_indices=topk_indices,
            softmax_scale=head_dim**-0.5,
            mask=mask,
            query_chunk_size=2,
            use_gather=True,
        )

    recompute_output = torch_checkpoint.checkpoint(
        _compute_recompute_output,
        query,
        key,
        value,
        use_reentrant=False,
    )
    recompute_output.sum().backward()

    torch.testing.assert_close(recompute_output, normal_output)
    torch.testing.assert_close(query.grad, normal_grads[0])
    torch.testing.assert_close(key.grad, normal_grads[1])
    torch.testing.assert_close(value.grad, normal_grads[2])


def test_fused_qk_topk_naive_caps_topk_by_key_length():
    torch.manual_seed(123)

    q = torch.randn(2, 1, 4, 8, dtype=torch.float32)
    k = torch.randn(5, 1, 8, dtype=torch.float32)
    weights = torch.randn(2, 1, 4, dtype=torch.float32)

    _, topk_indices = fused_qk_topk_naive(q=q, k=k, weights=weights, index_topk=4)

    assert topk_indices.shape == (1, 2, 4)
    assert torch.all((topk_indices >= 0) & (topk_indices < 5))


def test_fused_qk_topk_chunked_matches_dense_reference():
    torch.manual_seed(123)

    seqlen_q = 7
    seqlen_k = 9
    batch_size = 2
    num_index_heads = 4
    head_dim = 8
    topk = 3

    q = torch.randn(seqlen_q, batch_size, num_index_heads, head_dim, dtype=torch.float32)
    k = torch.randn(seqlen_k, batch_size, head_dim, dtype=torch.float32)
    weights = torch.randn(seqlen_q, batch_size, num_index_heads, dtype=torch.float32)
    mask = torch.zeros(batch_size, seqlen_q, seqlen_k, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    dense_scores, dense_indices = fused_qk_topk_naive(
        q=q,
        k=k,
        weights=weights,
        index_topk=topk,
        mask=mask,
    )
    chunked_scores, chunked_indices = fused_qk_topk_chunked(
        q=q,
        k=k,
        weights=weights,
        index_topk=topk,
        mask=mask,
        key_chunk_size=4,
    )

    expected_chunked_scores = dense_scores.gather(-1, chunked_indices)
    torch.testing.assert_close(chunked_scores, expected_chunked_scores)
    torch.testing.assert_close(
        torch.sort(chunked_scores, dim=-1).values,
        torch.sort(dense_scores.gather(-1, dense_indices), dim=-1).values,
    )


def test_fused_qk_topk_chunked_recompute_matches_normal():
    torch.manual_seed(123)

    seqlen_q = 7
    seqlen_k = 9
    batch_size = 2
    num_index_heads = 4
    head_dim = 8
    topk = 3

    q = torch.randn(
        seqlen_q, batch_size, num_index_heads, head_dim, dtype=torch.float32, requires_grad=True
    )
    k = torch.randn(seqlen_k, batch_size, head_dim, dtype=torch.float32, requires_grad=True)
    weights = torch.randn(
        seqlen_q, batch_size, num_index_heads, dtype=torch.float32, requires_grad=True
    )
    mask = torch.zeros(batch_size, seqlen_q, seqlen_k, dtype=torch.float32)
    mask[:, :, -1] = float("-inf")

    normal_scores, normal_indices = fused_qk_topk_chunked(
        q=q,
        k=k,
        weights=weights,
        index_topk=topk,
        mask=mask,
        key_chunk_size=4,
    )
    normal_scores.sum().backward()
    normal_grads = (q.grad.clone(), k.grad.clone(), weights.grad.clone())

    q.grad = None
    k.grad = None
    weights.grad = None

    def _compute_chunked_topk(q_tensor, k_tensor, weights_tensor):
        return fused_qk_topk_chunked(
            q=q_tensor,
            k=k_tensor,
            weights=weights_tensor,
            index_topk=topk,
            mask=mask,
            key_chunk_size=4,
        )

    recompute_scores, recompute_indices = torch_checkpoint.checkpoint(
        _compute_chunked_topk,
        q,
        k,
        weights,
        use_reentrant=False,
    )
    recompute_scores.sum().backward()

    torch.testing.assert_close(recompute_scores, normal_scores)
    torch.testing.assert_close(recompute_indices, normal_indices)
    torch.testing.assert_close(q.grad, normal_grads[0])
    torch.testing.assert_close(k.grad, normal_grads[1])
    torch.testing.assert_close(weights.grad, normal_grads[2])


def test_compute_gqa_dsa_indexer_loss_recompute_matches_normal():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    def _compute_loss(index_scores_tensor):
        return compute_gqa_dsa_indexer_loss(
            index_scores=index_scores_tensor,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=head_dim**-0.5,
            loss_coeff=0.7,
            sparse_loss=True,
            pg_collection=pg_collection,
        )

    normal_loss = _compute_loss(index_scores)
    normal_loss.backward()
    normal_grad = index_scores.grad.clone()

    index_scores.grad = None

    recompute_loss = torch_checkpoint.checkpoint(
        _compute_loss,
        index_scores,
        use_reentrant=False,
    )
    recompute_loss.backward()

    torch.testing.assert_close(recompute_loss, normal_loss)
    torch.testing.assert_close(index_scores.grad, normal_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_recompute_matches_normal():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    def _compute_loss(index_scores_tensor):
        return compute_gqa_dsa_indexer_loss(
            index_scores=index_scores_tensor,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=head_dim**-0.5,
            loss_coeff=0.7,
            sparse_loss=True,
            pg_collection=pg_collection,
            sparse_loss_use_topk_only=True,
        )

    normal_loss = _compute_loss(index_scores)
    normal_loss.backward()
    normal_grad = index_scores.grad.clone()

    index_scores.grad = None

    recompute_loss = torch_checkpoint.checkpoint(
        _compute_loss,
        index_scores,
        use_reentrant=False,
    )
    recompute_loss.backward()

    torch.testing.assert_close(recompute_loss, normal_loss)
    torch.testing.assert_close(index_scores.grad, normal_grad)


def test_compute_gqa_dsa_indexer_loss_sparse_topk_only_chunked_recompute_matches_normal():
    torch.manual_seed(123)

    batch_size = 2
    seqlen = 8
    num_heads = 8
    num_query_groups = 2
    head_dim = 16
    topk = 4

    index_scores = _causal_index_scores(
        torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32, requires_grad=True)
    )
    topk_indices = index_scores.detach().topk(topk, dim=-1).indices
    query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seqlen, batch_size, num_query_groups, head_dim, dtype=torch.float32)
    pg_collection = _DummyPGCollection()

    def _compute_loss(index_scores_tensor):
        return compute_gqa_dsa_indexer_loss(
            index_scores=index_scores_tensor,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=head_dim**-0.5,
            loss_coeff=0.7,
            sparse_loss=True,
            pg_collection=pg_collection,
            sparse_loss_use_topk_only=True,
            query_chunk_size=3,
        )

    normal_loss = _compute_loss(index_scores)
    normal_loss.backward()
    normal_grad = index_scores.grad.clone()

    index_scores.grad = None

    recompute_loss = torch_checkpoint.checkpoint(
        _compute_loss,
        index_scores,
        use_reentrant=False,
    )
    recompute_loss.backward()

    torch.testing.assert_close(recompute_loss, normal_loss)
    torch.testing.assert_close(index_scores.grad, normal_grad)


def test_build_shifted_causal_mask_respects_query_offset():
    mask = _build_shifted_causal_mask(query_length=2, key_length=5, query_start_position=3, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(mask, expected)
