# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sparse all-to-all key/value exchange for DSA-over-GQA context parallelism.

The all-gather path moves every rank's K and V to every other rank, so a rank can run the
indexer and attention against complete keys. That is simple and makes the top-k a genuinely
global selection, but it moves the whole sequence regardless of how little of it the top-k
actually selects.

This module moves only the selected keys. After routing picks ``topk_indices`` for a query
tile, a rank knows exactly which global key positions it needs; it asks each owner for that
owner's share and receives just those rows. The result is a drop-in substitution for the pair
the downstream code already consumes:

    (full key, global topk_indices)  ->  (received rows, remapped topk_indices)

Both the PyTorch and Triton backends gather ``key[index]`` internally and neither cares that
``key`` spans the whole sequence, so they work unchanged once the indices address the received
buffer instead. Causality is the exception: masks compare indices against global positions, so
the *global* indices must be kept for that and only the gather may use the remapped ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    zigzag_cp_owner_and_local_index,
)


@dataclass(frozen=True)
class TileExchangePlan:
    """Addressing for one query tile's key exchange.

    Attributes:
        send_counts: Rows this rank requests from each owner, indexed by rank.
        recv_counts: Rows each rank requests from this one, indexed by rank.
        requested_local: Local row indices into this rank's K/V that peers asked for,
            concatenated in rank order and split by ``recv_counts``.
        remap: Lookup from global key position to its row in the received buffer. Shaped
            ``[seq_len]``; only the positions this tile selected are meaningful.
        total_recv: Number of rows this rank receives, i.e. the size of the exchanged buffer.
    """

    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    requested_local: torch.Tensor
    remap: torch.Tensor
    total_recv: int


@dataclass(frozen=True)
class KVExchangeContext:
    """What a tile loop needs to swap keys instead of having been handed all of them.

    Carrying the global ``seq_len`` explicitly matters: with an exchange the key tensor is this
    rank's shard, so routing can no longer infer the sequence length from ``key.size(0)``.

    Attributes:
        cp_group: The context-parallel process group.
        cp_size: Its size.
        seq_len: Global sequence length.
    """

    cp_group: object
    cp_size: int
    seq_len: int


def exchange_tile_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    context: Optional[KVExchangeContext],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[TileExchangePlan]]:
    """Fetch a tile's selected keys and values, or pass through when there is no exchange.

    Returns ``(key, value, gather_indices, plan)``. ``gather_indices`` addresses the returned
    tensors and is None when they are the originals; callers keep ``topk_indices`` for causal
    masking either way. The plan comes back so a backward can hand gradients on the returned
    buffers to :func:`return_row_grads`.

    Runs without autograd. The min-memory backward recomputes tiles and routes dK and dV by
    hand, so a graph here would be dead weight; see :func:`exchange_rows`.
    """
    if context is None:
        return key, value, None, None
    plan = plan_tile_exchange(topk_indices, context.seq_len, context.cp_size, context.cp_group)
    return (
        exchange_rows(key, plan, context.cp_group),
        exchange_rows(value, plan, context.cp_group),
        plan.remap[topk_indices],
        plan,
    )


def plan_tile_exchange(
    topk_indices: torch.Tensor, seq_len: int, cp_size: int, cp_group
) -> TileExchangePlan:
    """Work out who holds this tile's selected keys and arrange the swap.

    Deduplicates across the whole tile -- every batch row and every query in it -- because one
    key selected by many queries only has to cross the wire once.

    Args:
        topk_indices: Global key positions selected for this tile, ``[batch, query, topk]``.
        seq_len: Global sequence length.
        cp_size: Context-parallel world size.
        cp_group: The context-parallel process group.

    Returns:
        The plan the exchange and the index remapping both read.
    """
    device = topk_indices.device
    wanted = torch.unique(topk_indices.reshape(-1))
    owner, local_index = zigzag_cp_owner_and_local_index(wanted, seq_len, cp_size)

    # Group the requests by owner. unique() returns position order, which zigzag does not keep
    # aligned with rank order, so this permutation is what makes the counts describe contiguous
    # blocks the collective can split on.
    order = torch.argsort(owner, stable=True)
    wanted_by_owner = wanted[order]
    local_by_owner = local_index[order]
    send_counts = torch.bincount(owner, minlength=cp_size).to(torch.int64)

    # Tell each owner how much to expect before sending indices of unknown length.
    recv_counts = torch.empty_like(send_counts)
    torch.distributed.all_to_all_single(recv_counts, send_counts, group=cp_group)

    send_split = send_counts.tolist()
    recv_split = recv_counts.tolist()
    requested_local = torch.empty(int(sum(recv_split)), dtype=torch.int64, device=device)
    torch.distributed.all_to_all_single(
        requested_local,
        local_by_owner.contiguous(),
        output_split_sizes=recv_split,
        input_split_sizes=send_split,
        group=cp_group,
    )

    # Replies arrive in the order the requests left, so row i of the received buffer is
    # wanted_by_owner[i]. Inverting that gives the remap the gather needs.
    remap = torch.zeros(seq_len, dtype=torch.int64, device=device)
    remap[wanted_by_owner] = torch.arange(wanted_by_owner.numel(), dtype=torch.int64, device=device)

    return TileExchangePlan(
        send_counts=send_counts,
        recv_counts=recv_counts,
        requested_local=requested_local,
        remap=remap,
        total_recv=int(wanted_by_owner.numel()),
    )


def exchange_rows(local_rows: torch.Tensor, plan: TileExchangePlan, cp_group) -> torch.Tensor:
    """Serve peers the rows they asked for and collect the rows this rank asked for.

    The plain forward half, with no autograd. The min-memory backend recomputes tiles inside a
    hand-written ``backward``, where building graph nodes would be dead weight, so the exchange
    has to be callable without one. :class:`_AllToAllSelectedRows` wraps this for callers that
    do want autograd.
    """
    served = local_rows.index_select(0, plan.requested_local)
    received = torch.empty(
        (plan.total_recv, *local_rows.shape[1:]), dtype=local_rows.dtype, device=local_rows.device
    )
    torch.distributed.all_to_all_single(
        received,
        served.contiguous(),
        output_split_sizes=plan.send_counts.tolist(),
        input_split_sizes=plan.recv_counts.tolist(),
        group=cp_group,
    )
    return received


def return_row_grads(
    grad_received: torch.Tensor, plan: TileExchangePlan, cp_group, local_shape
) -> torch.Tensor:
    """Route each received row's gradient home and accumulate it into the owner's shard.

    The plain backward half. Accumulation is an ``index_add_`` rather than an assignment
    because one key is typically selected by many queries on many ranks, and every one of them
    contributes. The all-gather path inherits this from the reduce-scatter in
    ``gather_from_sequence_parallel_region``; here it is explicit, and an assignment would give
    finite but wrong dK with no error.
    """
    grad_served = torch.empty(
        (plan.requested_local.numel(), *grad_received.shape[1:]),
        dtype=grad_received.dtype,
        device=grad_received.device,
    )
    torch.distributed.all_to_all_single(
        grad_served,
        grad_received.contiguous(),
        output_split_sizes=plan.recv_counts.tolist(),
        input_split_sizes=plan.send_counts.tolist(),
        group=cp_group,
    )
    grad_local = torch.zeros(local_shape, dtype=grad_received.dtype, device=grad_received.device)
    grad_local.index_add_(0, plan.requested_local, grad_served)
    return grad_local


class _AllToAllSelectedRows(torch.autograd.Function):
    """Serve peers the K (or V) rows they asked for, and route the gradient back.

    Forward gathers the requested rows out of the local shard and exchanges them. Backward has
    to undo both halves: send each received row's gradient back to the rank that asked for it,
    then accumulate into the local shard. Accumulation has to be an ``index_add_`` rather than
    an assignment because a single key is typically selected by many queries on many ranks, and
    every one of those contributes to its gradient. The all-gather path gets this for free from
    ``gather_from_sequence_parallel_region``'s reduce-scatter; here it is ours to do, and
    getting it wrong yields finite but wrong dK with no error.
    """

    @staticmethod
    def forward(ctx, local_rows: torch.Tensor, plan: TileExchangePlan, cp_group):
        """Exchange the selected rows, returning this rank's received buffer."""
        ctx.plan = plan
        ctx.cp_group = cp_group
        ctx.local_shape = local_rows.shape
        return exchange_rows(local_rows, plan, cp_group)

    @staticmethod
    def backward(ctx, grad_received: torch.Tensor):
        """Return the gradient to each requesting rank and accumulate it at the owner."""
        return return_row_grads(grad_received, ctx.plan, ctx.cp_group, ctx.local_shape), None, None


def exchange_selected_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_len: int,
    cp_size: int,
    cp_group,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch this tile's selected keys and values from their owners.

    Args:
        key: This rank's key shard, ``[local_seq, batch, groups, head_dim]``.
        value: This rank's value shard, same shape.
        topk_indices: Global key positions selected for the tile, ``[batch, query, topk]``.
        seq_len: Global sequence length.
        cp_size: Context-parallel world size.
        cp_group: The context-parallel process group.

    Returns:
        ``(received_key, received_value, remapped_indices)``. The first two are indexed by the
        third exactly as the full key and global indices were. The caller must keep the original
        ``topk_indices`` for causal masking, which compares against global positions.
    """
    plan = plan_tile_exchange(topk_indices, seq_len, cp_size, cp_group)
    received_key = _AllToAllSelectedRows.apply(key, plan, cp_group)
    received_value = _AllToAllSelectedRows.apply(value, plan, cp_group)
    return received_key, received_value, plan.remap[topk_indices]
