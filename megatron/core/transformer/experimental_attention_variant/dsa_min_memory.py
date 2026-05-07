# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Minimum-activation DSA-GQA training path.

This module deliberately keeps the implementation in tile-local PyTorch ops.  The public backend
name is the stable switch point for a future Triton kernel replacement; the contract this file
enforces today is that no full DSA routing scores, top-k tensors, sparse masks, selected K/V, or
attention probabilities are saved across the forward/backward boundary.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.models.common.embeddings.rope_utils import _rotate_half
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_get_concentration_factor,
    _yarn_linear_ramp_mask,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation


def _module_weight(module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise RuntimeError(f"{module.__class__.__name__} does not expose a weight tensor.")
    return weight


def _module_bias(module, like: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    bias = getattr(module, "bias", None)
    if bias is None:
        return like.new_empty((0,)), False
    return bias, True


def _default_query_chunk_size(query_length: int) -> int:
    return min(query_length, 128)


def _default_key_chunk_size(key_length: int) -> int:
    return min(key_length, 1024)


def _default_topk_score_chunk_size(topk: int) -> int:
    return min(topk, 128)


def _chunk_size(config_value: Optional[int], default_value: int, maximum: int) -> int:
    if config_value is None or config_value <= 0:
        return min(default_value, maximum)
    return min(config_value, maximum)


def _default_rotary_interleaved(rotary_pos_emb) -> bool:
    return getattr(rotary_pos_emb, "rotary_interleaved", False)


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight, None)


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x, (x.size(-1),), weight, bias if has_bias else None, eps)


def _rope_inv_freq_and_mscale(rotary_pos_emb, device: torch.device) -> Tuple[torch.Tensor, float]:
    if isinstance(rotary_pos_emb, YarnRotaryEmbedding):
        if rotary_pos_emb.inv_freq_extra.device.type == "cpu":
            rotary_pos_emb.inv_freq_extra = rotary_pos_emb.inv_freq_extra.to(device=device)
        if rotary_pos_emb.inv_freq_inter.device.type == "cpu":
            rotary_pos_emb.inv_freq_inter = rotary_pos_emb.inv_freq_inter.to(device=device)

        low, high = _yarn_find_correction_range(
            rotary_pos_emb.beta_fast,
            rotary_pos_emb.beta_slow,
            rotary_pos_emb.dim,
            rotary_pos_emb.rotary_base,
            rotary_pos_emb.original_max_position_embeddings,
            rotary_pos_emb.correction_range_round_to_int,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, rotary_pos_emb.dim // 2, device=rotary_pos_emb.inv_freq_extra.device
        ).to(dtype=torch.float32)
        inv_freq = (
            rotary_pos_emb.inv_freq_inter * (1 - inv_freq_mask)
            + rotary_pos_emb.inv_freq_extra * inv_freq_mask
        )
        mscale = _yarn_get_concentration_factor(
            rotary_pos_emb.scaling_factor, rotary_pos_emb.mscale, rotary_pos_emb.mscale_all_dim
        )
        return inv_freq, mscale

    if rotary_pos_emb.inv_freq.device.type == "cpu":
        rotary_pos_emb.inv_freq = rotary_pos_emb.inv_freq.to(device=device)
    return rotary_pos_emb.inv_freq, 1.0


def _apply_rope_at_positions(
    x: torch.Tensor,
    positions: torch.Tensor,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
) -> torch.Tensor:
    if rotary_pos_emb is None or index_rotary_dim == 0:
        return x

    x_nope, x_pe = torch.split(x, [index_head_dim - index_rotary_dim, index_rotary_dim], dim=-1)
    inv_freq, mscale = _rope_inv_freq_and_mscale(rotary_pos_emb, x.device)
    inv_freq = inv_freq[: index_rotary_dim // 2]

    positions = positions.to(device=inv_freq.device, dtype=inv_freq.dtype)
    interpolation_factor = getattr(rotary_pos_emb, "seq_len_interpolation_factor", None)
    if interpolation_factor is not None and not isinstance(rotary_pos_emb, YarnRotaryEmbedding):
        positions = positions * (1 / interpolation_factor)
    freqs = positions.unsqueeze(-1) * inv_freq
    if not getattr(rotary_pos_emb, "rotary_interleaved", False):
        freqs = torch.cat((freqs, freqs), dim=-1)
    else:
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=-2)
    while freqs.dim() < x_pe.dim():
        freqs = freqs.unsqueeze(-2)

    cos = (torch.cos(freqs) * mscale).to(dtype=x_pe.dtype, device=x_pe.device)
    sin = (torch.sin(freqs) * mscale).to(dtype=x_pe.dtype, device=x_pe.device)
    x_pe = x_pe * cos + _rotate_half(x_pe, rotary_interleaved) * sin
    return torch.cat((x_nope, x_pe), dim=-1)


def _project_q_index_tile(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    linear_q_weight: torch.Tensor,
    linear_weights_weight: torch.Tensor,
    index_n_heads: int,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_tile = hidden_states[q_start:q_end]
    q = _linear(hidden_tile, linear_q_weight)
    q = q.reshape(q_end - q_start, hidden_states.size(1), index_n_heads, index_head_dim)
    if use_indexer_rope:
        positions = torch.arange(q_start, q_end, device=q.device, dtype=torch.long)
        q = _apply_rope_at_positions(
            q, positions, index_head_dim, index_rotary_dim, rotary_pos_emb, rotary_interleaved
        )
    if use_hadamard:
        q = rotate_activation(q)

    weights = _linear(hidden_tile, linear_weights_weight)
    weights = weights * (index_n_heads**-0.5) * (index_head_dim**-0.5)
    return q, weights


def _project_k_index_block(
    hidden_states: torch.Tensor,
    k_start: int,
    k_end: int,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> torch.Tensor:
    k = _linear(hidden_states[k_start:k_end], linear_k_weight)
    k = _layer_norm(k, k_norm_weight, k_norm_bias, has_k_norm_bias, k_norm_eps)
    if use_indexer_rope:
        k = k.reshape(k_end - k_start, hidden_states.size(1), 1, index_head_dim)
        positions = torch.arange(k_start, k_end, device=k.device, dtype=torch.long)
        k = _apply_rope_at_positions(
            k, positions, index_head_dim, index_rotary_dim, rotary_pos_emb, rotary_interleaved
        )
        k = k.reshape(k_end - k_start, hidden_states.size(1), index_head_dim)
    if use_hadamard:
        k = rotate_activation(k)
    return k


def _project_selected_k_index(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> torch.Tensor:
    batch_size, query_length, topk = topk_indices.shape
    hidden_by_batch = hidden_states.permute(1, 0, 2)
    batch_index = torch.arange(batch_size, device=topk_indices.device).view(batch_size, 1, 1)
    selected_hidden = hidden_by_batch[batch_index, topk_indices]
    k = _linear(selected_hidden, linear_k_weight)
    k = _layer_norm(k, k_norm_weight, k_norm_bias, has_k_norm_bias, k_norm_eps)
    if use_indexer_rope:
        k = _apply_rope_at_positions(
            k,
            topk_indices,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
        )
    if use_hadamard:
        k = rotate_activation(k)
    return k


def _index_scores_for_block(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    k_index: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("qbhd,tbd->bqht", q_index.float(), k_index.float())
    scores = torch.relu(scores)
    scores = scores * weights.permute(1, 0, 2).unsqueeze(-1).float()
    return scores.sum(dim=2)


def _index_scores_for_selected(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
) -> torch.Tensor:
    q_index = q_index.permute(1, 0, 2, 3)
    weights = weights.permute(1, 0, 2)
    scores = torch.einsum("bqhd,bqkd->bqhk", q_index.float(), selected_k_index.float())
    scores = torch.relu(scores)
    scores = scores * weights.unsqueeze(-1).float()
    return scores.sum(dim=2)


def _causal_invalid_mask(
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(q_start, q_end, device=device, dtype=torch.long)
    key_positions = torch.arange(k_start, k_end, device=device, dtype=torch.long)
    return key_positions.view(1, k_end - k_start) > query_positions.view(q_end - q_start, 1)


def _selected_causal_invalid_mask(
    topk_indices: torch.Tensor,
    q_start: int,
) -> torch.Tensor:
    query_positions = torch.arange(
        q_start,
        q_start + topk_indices.size(1),
        device=topk_indices.device,
        dtype=topk_indices.dtype,
    )
    return topk_indices > query_positions.view(1, topk_indices.size(1), 1)


def _merge_topk(
    running_scores: Optional[torch.Tensor],
    running_indices: Optional[torch.Tensor],
    block_scores: torch.Tensor,
    block_indices: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if running_scores is None or running_indices is None:
        return block_scores, block_indices
    merged_scores = torch.cat((running_scores, block_scores), dim=-1)
    merged_indices = torch.cat((running_indices, block_indices), dim=-1)
    keep = merged_scores.topk(min(topk, merged_scores.size(-1)), dim=-1).indices
    return torch.gather(merged_scores, -1, keep), torch.gather(merged_indices, -1, keep)


def _topk_index_tile(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    linear_q_weight: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    linear_weights_weight: torch.Tensor,
    k_norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    key_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_index, weights = _project_q_index_tile(
        hidden_states,
        q_start,
        q_end,
        linear_q_weight,
        linear_weights_weight,
        index_n_heads,
        index_head_dim,
        index_rotary_dim,
        rotary_pos_emb,
        rotary_interleaved,
        use_indexer_rope,
        use_hadamard,
    )
    topk = min(index_topk, hidden_states.size(0))
    running_scores = None
    running_indices = None
    for k_start in range(0, hidden_states.size(0), key_chunk_size):
        k_end = min(k_start + key_chunk_size, hidden_states.size(0))
        k_index = _project_k_index_block(
            hidden_states,
            k_start,
            k_end,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            has_k_norm_bias,
            k_norm_eps,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
            use_indexer_rope,
            use_hadamard,
        )
        block_scores = _index_scores_for_block(q_index, weights, k_index)
        invalid = _causal_invalid_mask(q_start, q_end, k_start, k_end, block_scores.device)
        block_scores = block_scores.masked_fill(invalid.unsqueeze(0), float("-inf"))
        block_topk = min(topk, k_end - k_start)
        block_scores, block_indices = block_scores.topk(block_topk, dim=-1)
        block_indices = block_indices + k_start
        running_scores, running_indices = _merge_topk(
            running_scores, running_indices, block_scores, block_indices, topk
        )
    return running_scores, running_indices, q_index, weights


def _gather_selected_kv(
    tensor: torch.Tensor,
    group_idx: int,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    tensor = tensor[:, :, group_idx, :].permute(1, 0, 2)
    batch_size = topk_indices.size(0)
    batch_index = torch.arange(batch_size, device=topk_indices.device).view(batch_size, 1, 1)
    return tensor[batch_index, topk_indices]


def _sparse_attention_tile(
    query_tile: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int,
) -> torch.Tensor:
    query_length, batch_size, num_query_heads, head_dim = query_tile.shape
    num_query_groups = key.size(2)
    repeat_factor = num_query_heads // num_query_groups
    value_head_dim = value.size(-1)
    output = value.new_empty(
        (query_length, batch_size, num_query_heads, value_head_dim), dtype=value.dtype
    )
    selected_invalid = _selected_causal_invalid_mask(topk_indices, q_start).unsqueeze(1)

    for group_idx in range(num_query_groups):
        head_start = group_idx * repeat_factor
        head_end = head_start + repeat_factor
        query_group = query_tile[:, :, head_start:head_end, :].permute(1, 2, 0, 3)
        selected_key = _gather_selected_kv(key, group_idx, topk_indices)
        selected_value = _gather_selected_kv(value, group_idx, topk_indices)
        scores = (
            torch.einsum("brqd,bqkd->brqk", query_group.float(), selected_key.float())
            * softmax_scale
        )
        scores = scores.masked_fill(selected_invalid, float("-inf"))
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        group_output = torch.einsum(
            "brqk,bqkd->brqd", probs.to(selected_value.dtype), selected_value
        )
        output[:, :, head_start:head_end, :] = group_output.permute(2, 0, 1, 3)

    return output


def _teacher_scores_tile(
    query_tile: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    _, batch_size, num_query_heads, _ = query_tile.shape
    num_query_groups = key.size(2)
    repeat_factor = num_query_heads // num_query_groups
    teacher = query_tile.new_zeros(
        (batch_size, topk_indices.size(1), topk_indices.size(2)), dtype=torch.float32
    )
    selected_invalid = _selected_causal_invalid_mask(topk_indices, q_start).unsqueeze(1)

    for group_idx in range(num_query_groups):
        head_start = group_idx * repeat_factor
        head_end = head_start + repeat_factor
        query_group = query_tile[:, :, head_start:head_end, :].permute(1, 2, 0, 3)
        selected_key = _gather_selected_kv(key, group_idx, topk_indices)
        scores = (
            torch.einsum("brqd,bqkd->brqk", query_group.float(), selected_key.float())
            * softmax_scale
        )
        scores = scores.masked_fill(selected_invalid, float("-inf"))
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        teacher = teacher + probs.sum(dim=1)

    if pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(teacher.contiguous(), group=pg_collection.tp)
    return teacher / teacher.sum(dim=-1, keepdim=True)


def _indexer_loss_tile(
    selected_index_scores: torch.Tensor,
    query_tile: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    total_positions: int,
    q_start: int,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    teacher = _teacher_scores_tile(
        query_tile.detach(), key.detach(), topk_indices, softmax_scale, q_start, pg_collection
    )
    student = torch.nn.functional.softmax(selected_index_scores, dim=-1, dtype=torch.float32)
    kl = teacher * (torch.log(teacher + 1e-10) - torch.log(student + 1e-10))
    return kl.sum() * (loss_coeff / total_positions)


def _selected_index_scores_tile(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    topk_indices: torch.Tensor,
    q_index: torch.Tensor,
    weights: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> torch.Tensor:
    selected_k_index = _project_selected_k_index(
        hidden_states,
        topk_indices,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        has_k_norm_bias,
        k_norm_eps,
        index_head_dim,
        index_rotary_dim,
        rotary_pos_emb,
        rotary_interleaved,
        use_indexer_rope,
        use_hadamard,
    )
    selected_scores = _index_scores_for_selected(q_index, weights, selected_k_index)
    invalid = _selected_causal_invalid_mask(topk_indices, q_start)
    return selected_scores.masked_fill(invalid, float("-inf"))


def _selected_index_scores_tile_chunked(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    topk_indices: torch.Tensor,
    q_index: torch.Tensor,
    weights: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    topk_score_chunk_size: int,
) -> torch.Tensor:
    if topk_score_chunk_size <= 0 or topk_score_chunk_size >= topk_indices.size(-1):
        return _selected_index_scores_tile(
            hidden_states,
            q_start,
            q_end,
            topk_indices,
            q_index,
            weights,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            has_k_norm_bias,
            k_norm_eps,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
            use_indexer_rope,
            use_hadamard,
        )

    score_chunks = []
    for topk_start in range(0, topk_indices.size(-1), topk_score_chunk_size):
        topk_end = min(topk_start + topk_score_chunk_size, topk_indices.size(-1))
        score_chunks.append(
            _selected_index_scores_tile(
                hidden_states,
                q_start,
                q_end,
                topk_indices[..., topk_start:topk_end],
                q_index,
                weights,
                linear_k_weight,
                k_norm_weight,
                k_norm_bias,
                has_k_norm_bias,
                k_norm_eps,
                index_head_dim,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                use_hadamard,
            )
        )
    return torch.cat(score_chunks, dim=-1)


def _forward_min_memory_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hidden_states: torch.Tensor,
    linear_q_weight: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    linear_weights_weight: torch.Tensor,
    k_norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    use_indexer_rope: bool,
    use_hadamard: bool,
    softmax_scale: float,
    loss_coeff: float,
    query_chunk_size: int,
    key_chunk_size: int,
    pg_collection: ProcessGroupCollection,
    compute_loss: bool,
    rotary_interleaved: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sq, batch_size, num_query_heads, _ = query.shape
    output = value.new_empty((sq, batch_size, num_query_heads, value.size(-1)))
    indexer_loss = query.new_zeros((), dtype=torch.float32)
    total_positions = batch_size * sq
    if rotary_interleaved is None:
        rotary_interleaved = _default_rotary_interleaved(rotary_pos_emb)

    for q_start in range(0, sq, query_chunk_size):
        q_end = min(q_start + query_chunk_size, sq)
        _, topk_indices, q_index, weights = _topk_index_tile(
            hidden_states,
            q_start,
            q_end,
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            has_k_norm_bias,
            linear_weights_weight,
            k_norm_eps,
            index_n_heads,
            index_head_dim,
            index_topk,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
            use_indexer_rope,
            use_hadamard,
            key_chunk_size,
        )
        query_tile = query[q_start:q_end]
        output[q_start:q_end] = _sparse_attention_tile(
            query_tile, key, value, topk_indices, softmax_scale, q_start
        )
        if compute_loss and loss_coeff > 0:
            selected_index_scores = _selected_index_scores_tile_chunked(
                hidden_states,
                q_start,
                q_end,
                topk_indices,
                q_index,
                weights,
                linear_k_weight,
                k_norm_weight,
                k_norm_bias,
                has_k_norm_bias,
                k_norm_eps,
                index_head_dim,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                use_hadamard,
                _default_topk_score_chunk_size(topk_indices.size(-1)),
            )
            indexer_loss = indexer_loss + _indexer_loss_tile(
                selected_index_scores,
                query_tile,
                key,
                topk_indices,
                softmax_scale,
                loss_coeff,
                total_positions,
                q_start,
                pg_collection,
            )

    return output.reshape(sq, batch_size, num_query_heads * value.size(-1)), indexer_loss


class DSAMinMemoryGQAFn(torch.autograd.Function):
    """Recompute DSA-GQA routing, sparse attention, and sparse KL one query tile at a time."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden_states: torch.Tensor,
        linear_q_weight: torch.Tensor,
        linear_k_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        linear_weights_weight: torch.Tensor,
        has_k_norm_bias: bool,
        k_norm_eps: float,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        index_rotary_dim: int,
        rotary_pos_emb,
        use_indexer_rope: bool,
        use_hadamard: bool,
        softmax_scale: float,
        loss_coeff: float,
        query_chunk_size: int,
        key_chunk_size: int,
        pg_collection: ProcessGroupCollection,
        rotary_interleaved: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output, indexer_loss = _forward_min_memory_impl(
                query,
                key,
                value,
                hidden_states,
                linear_q_weight,
                linear_k_weight,
                k_norm_weight,
                k_norm_bias,
                has_k_norm_bias,
                linear_weights_weight,
                k_norm_eps,
                index_n_heads,
                index_head_dim,
                index_topk,
                index_rotary_dim,
                rotary_pos_emb,
                use_indexer_rope,
                use_hadamard,
                softmax_scale,
                loss_coeff,
                query_chunk_size,
                key_chunk_size,
                pg_collection,
                compute_loss=True,
                rotary_interleaved=rotary_interleaved,
            )

        ctx.save_for_backward(
            query,
            key,
            value,
            hidden_states,
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        )
        ctx.has_k_norm_bias = has_k_norm_bias
        ctx.k_norm_eps = k_norm_eps
        ctx.index_n_heads = index_n_heads
        ctx.index_head_dim = index_head_dim
        ctx.index_topk = index_topk
        ctx.index_rotary_dim = index_rotary_dim
        ctx.rotary_pos_emb = rotary_pos_emb
        ctx.rotary_interleaved = (
            _default_rotary_interleaved(rotary_pos_emb)
            if rotary_interleaved is None
            else rotary_interleaved
        )
        ctx.use_indexer_rope = use_indexer_rope
        ctx.use_hadamard = use_hadamard
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.query_chunk_size = query_chunk_size
        ctx.key_chunk_size = key_chunk_size
        ctx.pg_collection = pg_collection
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_indexer_loss: torch.Tensor):
        (
            query,
            key,
            value,
            hidden_states,
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        ) = ctx.saved_tensors
        sq, batch_size, num_query_heads, _ = query.shape
        value_head_dim = value.size(-1)

        grad_output = grad_output.reshape(sq, batch_size, num_query_heads, value_head_dim)
        grad_query = torch.zeros_like(query) if ctx.needs_input_grad[0] else None
        grad_key = torch.zeros_like(key) if ctx.needs_input_grad[1] else None
        grad_value = torch.zeros_like(value) if ctx.needs_input_grad[2] else None
        grad_linear_q_weight = (
            torch.zeros_like(linear_q_weight) if ctx.needs_input_grad[4] else None
        )
        grad_linear_k_weight = (
            torch.zeros_like(linear_k_weight) if ctx.needs_input_grad[5] else None
        )
        grad_k_norm_weight = torch.zeros_like(k_norm_weight) if ctx.needs_input_grad[6] else None
        grad_k_norm_bias = torch.zeros_like(k_norm_bias) if ctx.needs_input_grad[7] else None
        grad_linear_weights_weight = (
            torch.zeros_like(linear_weights_weight) if ctx.needs_input_grad[8] else None
        )

        total_positions = batch_size * sq
        compute_loss_grads = (
            grad_indexer_loss is not None
            and ctx.loss_coeff > 0
            and (
                ctx.needs_input_grad[4]
                or ctx.needs_input_grad[5]
                or ctx.needs_input_grad[6]
                or ctx.needs_input_grad[7]
                or ctx.needs_input_grad[8]
            )
        )

        for q_start in range(0, sq, ctx.query_chunk_size):
            q_end = min(q_start + ctx.query_chunk_size, sq)

            with torch.no_grad():
                _, topk_indices, _, _ = _topk_index_tile(
                    hidden_states,
                    q_start,
                    q_end,
                    linear_q_weight,
                    linear_k_weight,
                    k_norm_weight,
                    k_norm_bias,
                    ctx.has_k_norm_bias,
                    linear_weights_weight,
                    ctx.k_norm_eps,
                    ctx.index_n_heads,
                    ctx.index_head_dim,
                    ctx.index_topk,
                    ctx.index_rotary_dim,
                    ctx.rotary_pos_emb,
                    ctx.rotary_interleaved,
                    ctx.use_indexer_rope,
                    ctx.use_hadamard,
                    ctx.key_chunk_size,
                )

            attention_inputs = []
            query_tile = query[q_start:q_end].detach().requires_grad_(ctx.needs_input_grad[0])
            key_leaf = key.detach().requires_grad_(ctx.needs_input_grad[1])
            value_leaf = value.detach().requires_grad_(ctx.needs_input_grad[2])
            if ctx.needs_input_grad[0]:
                attention_inputs.append(query_tile)
            if ctx.needs_input_grad[1]:
                attention_inputs.append(key_leaf)
            if ctx.needs_input_grad[2]:
                attention_inputs.append(value_leaf)

            if attention_inputs:
                with torch.enable_grad():
                    output_tile = _sparse_attention_tile(
                        query_tile,
                        key_leaf,
                        value_leaf,
                        topk_indices,
                        ctx.softmax_scale,
                        q_start,
                    )
                attention_grads = torch.autograd.grad(
                    output_tile,
                    attention_inputs,
                    grad_outputs=grad_output[q_start:q_end],
                    retain_graph=False,
                    allow_unused=True,
                )
                grad_iter = iter(attention_grads)
                if ctx.needs_input_grad[0]:
                    grad = next(grad_iter)
                    if grad is not None:
                        grad_query[q_start:q_end] = grad
                if ctx.needs_input_grad[1]:
                    grad = next(grad_iter)
                    if grad is not None:
                        grad_key.add_(grad)
                if ctx.needs_input_grad[2]:
                    grad = next(grad_iter)
                    if grad is not None:
                        grad_value.add_(grad)

            if compute_loss_grads:
                lq_weight = linear_q_weight.detach().requires_grad_(ctx.needs_input_grad[4])
                lk_weight = linear_k_weight.detach().requires_grad_(ctx.needs_input_grad[5])
                kn_weight = k_norm_weight.detach().requires_grad_(ctx.needs_input_grad[6])
                kn_bias = k_norm_bias.detach().requires_grad_(ctx.needs_input_grad[7])
                lw_weight = linear_weights_weight.detach().requires_grad_(ctx.needs_input_grad[8])
                loss_inputs = []
                if ctx.needs_input_grad[4]:
                    loss_inputs.append(lq_weight)
                if ctx.needs_input_grad[5]:
                    loss_inputs.append(lk_weight)
                if ctx.needs_input_grad[6]:
                    loss_inputs.append(kn_weight)
                if ctx.needs_input_grad[7]:
                    loss_inputs.append(kn_bias)
                if ctx.needs_input_grad[8]:
                    loss_inputs.append(lw_weight)

                with torch.no_grad():
                    q_index, weights = _project_q_index_tile(
                        hidden_states.detach(),
                        q_start,
                        q_end,
                        linear_q_weight,
                        linear_weights_weight,
                        ctx.index_n_heads,
                        ctx.index_head_dim,
                        ctx.index_rotary_dim,
                        ctx.rotary_pos_emb,
                        ctx.rotary_interleaved,
                        ctx.use_indexer_rope,
                        ctx.use_hadamard,
                    )
                    selected_scores = _selected_index_scores_tile_chunked(
                        hidden_states.detach(),
                        q_start,
                        q_end,
                        topk_indices,
                        q_index,
                        weights,
                        lk_weight,
                        kn_weight,
                        kn_bias,
                        ctx.has_k_norm_bias,
                        ctx.k_norm_eps,
                        ctx.index_head_dim,
                        ctx.index_rotary_dim,
                        ctx.rotary_pos_emb,
                        ctx.rotary_interleaved,
                        ctx.use_indexer_rope,
                        ctx.use_hadamard,
                        _default_topk_score_chunk_size(topk_indices.size(-1)),
                    )
                    teacher = _teacher_scores_tile(
                        query[q_start:q_end],
                        key,
                        topk_indices,
                        ctx.softmax_scale,
                        q_start,
                        ctx.pg_collection,
                    )
                    student = torch.nn.functional.softmax(
                        selected_scores, dim=-1, dtype=torch.float32
                    )
                    teacher_over_student = teacher * student / (student + 1e-10)
                    grad_selected_scores = student * teacher_over_student.sum(
                        dim=-1, keepdim=True
                    ) - teacher_over_student
                    grad_selected_scores = (
                        grad_selected_scores
                        * (ctx.loss_coeff / total_positions)
                        * grad_indexer_loss
                    )

                topk_score_chunk_size = _default_topk_score_chunk_size(topk_indices.size(-1))
                for topk_start in range(0, topk_indices.size(-1), topk_score_chunk_size):
                    topk_end = min(topk_start + topk_score_chunk_size, topk_indices.size(-1))
                    with torch.enable_grad():
                        q_index, weights = _project_q_index_tile(
                            hidden_states.detach(),
                            q_start,
                            q_end,
                            lq_weight,
                            lw_weight,
                            ctx.index_n_heads,
                            ctx.index_head_dim,
                            ctx.index_rotary_dim,
                            ctx.rotary_pos_emb,
                            ctx.rotary_interleaved,
                            ctx.use_indexer_rope,
                            ctx.use_hadamard,
                        )
                        selected_scores = _selected_index_scores_tile(
                            hidden_states.detach(),
                            q_start,
                            q_end,
                            topk_indices[..., topk_start:topk_end],
                            q_index,
                            weights,
                            lk_weight,
                            kn_weight,
                            kn_bias,
                            ctx.has_k_norm_bias,
                            ctx.k_norm_eps,
                            ctx.index_head_dim,
                            ctx.index_rotary_dim,
                            ctx.rotary_pos_emb,
                            ctx.rotary_interleaved,
                            ctx.use_indexer_rope,
                            ctx.use_hadamard,
                        )

                    loss_grads = torch.autograd.grad(
                        selected_scores,
                        loss_inputs,
                        grad_outputs=grad_selected_scores[..., topk_start:topk_end],
                        retain_graph=False,
                        allow_unused=True,
                    )
                    grad_iter = iter(loss_grads)
                    if ctx.needs_input_grad[4]:
                        grad = next(grad_iter)
                        if grad is not None:
                            grad_linear_q_weight.add_(grad)
                    if ctx.needs_input_grad[5]:
                        grad = next(grad_iter)
                        if grad is not None:
                            grad_linear_k_weight.add_(grad)
                    if ctx.needs_input_grad[6]:
                        grad = next(grad_iter)
                        if grad is not None:
                            grad_k_norm_weight.add_(grad)
                    if ctx.needs_input_grad[7]:
                        grad = next(grad_iter)
                        if grad is not None:
                            grad_k_norm_bias.add_(grad)
                    if ctx.needs_input_grad[8]:
                        grad = next(grad_iter)
                        if grad is not None:
                            grad_linear_weights_weight.add_(grad)

        return (
            grad_query,
            grad_key,
            grad_value,
            None,
            grad_linear_q_weight,
            grad_linear_k_weight,
            grad_k_norm_weight,
            grad_k_norm_bias,
            grad_linear_weights_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dsa_min_memory_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hidden_states: torch.Tensor,
    indexer,
    softmax_scale: float,
    loss_coeff: float,
    use_indexer_rope: bool,
    query_chunk_size: Optional[int],
    key_chunk_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the minimum-activation DSA-GQA training backend."""
    k_norm_bias, has_k_norm_bias = _module_bias(indexer.k_norm, query)
    return DSAMinMemoryGQAFn.apply(
        query,
        key,
        value,
        hidden_states,
        _module_weight(indexer.linear_q),
        _module_weight(indexer.linear_k),
        _module_weight(indexer.k_norm),
        k_norm_bias,
        _module_weight(indexer.linear_weights_proj),
        has_k_norm_bias,
        getattr(indexer.k_norm, "eps", indexer.config.layernorm_epsilon),
        indexer.index_n_heads,
        indexer.index_head_dim,
        indexer.index_topk,
        indexer.index_rotary_dim,
        indexer.rotary_pos_emb,
        use_indexer_rope,
        indexer.config.dsa_indexer_use_hadamard,
        softmax_scale,
        loss_coeff,
        _chunk_size(query_chunk_size, _default_query_chunk_size(query.size(0)), query.size(0)),
        _chunk_size(key_chunk_size, _default_key_chunk_size(key.size(0)), key.size(0)),
        indexer.pg_collection,
        getattr(indexer.config, "rotary_interleaved", False),
    )
