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

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch



# Set DSA_A2A_STATS=1 to have the exchange record how much it actually moves.
#
# This answers the question the design rests on. Moving "only the selected keys" is a saving
# only while the keys a tile selects are few; a tile holds many queries, each picks top-k, and
# the exchange sends the union. Once that union approaches the sequence, the exchange is an
# all-gather with extra bookkeeping, and no amount of tuning the bookkeeping will help -- the
# tile has to get smaller. Accumulating it costs nothing: total_recv is already a host-side
# int, so there is no device read here.
# Note the per-tile/per-layer trap when reading ``rows_if_all_gathered``: an all-gather runs
# once for a layer, while the exchange runs once per tile, so the two are only comparable
# after multiplying this by the tile count. A tile fetching 74% of what one all-gather moves
# is not a saving when the layer has two tiles -- it is half again as much traffic.
_STATS_ENABLED = os.environ.get("DSA_A2A_STATS") == "1"
_stats = {"tiles": 0, "rows_received": 0, "rows_if_all_gathered": 0}

# Wall-clock attribution, so "optimize the exchange" is aimed by measurement rather than by
# which part looks expensive. Events are recorded and only read at exit: querying one forces a
# synchronise, which is the cost this instrumentation exists to find.
_timed_regions: dict = {}


class _NullRegion:
    """What :func:`_TimedRegion` returns when stats are off: no allocation, no work.

    The timed regions sit inside the per-tile path, so when the instrumentation is disabled
    -- which is always, outside a diagnostic run -- it must cost nothing at all. A single
    shared instance is handed out rather than a new object per call.
    """

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_REGION = _NullRegion()


class _CudaTimedRegion:
    """Bracket a region with CUDA events and keep them for a later, offline read."""

    __slots__ = ("name", "start", "stop")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.stop.record()
        # Capped: the point is a representative sample, not a complete trace, and an
        # unbounded list of events is its own memory problem on a long run.
        bucket = _timed_regions.setdefault(self.name, [])
        if len(bucket) < 2000:
            bucket.append((self.start, self.stop))
        return False


def _TimedRegion(name: str):
    """A timed region when stats are on, and the shared no-op when they are not."""
    if _STATS_ENABLED and torch.cuda.is_available():
        return _CudaTimedRegion(name)
    return _NULL_REGION


def exchange_stats() -> dict:
    """Totals recorded since the last reset, empty unless ``DSA_A2A_STATS=1``."""
    return dict(_stats)


def reset_exchange_stats() -> None:
    """Zero the counters, so a caller can measure one step rather than the whole run."""
    for key in _stats:
        _stats[key] = 0


if _STATS_ENABLED:  # pragma: no cover - diagnostic path
    import atexit

    @atexit.register
    def _report_exchange_stats():
        """Print the totals at exit, so no training-loop hook is needed to see them."""
        if not _stats["tiles"]:
            return
        received, gathered = _stats["rows_received"], _stats["rows_if_all_gathered"]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(
            f"[dsa-a2a-stats rank {rank}] tiles={_stats['tiles']} "
            f"rows_received={received} rows_if_all_gathered_per_tile={gathered} "
            f"per_tile_ratio={received / max(1, gathered):.3f}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            for name, events in sorted(_timed_regions.items()):
                total = sum(start.elapsed_time(stop) for start, stop in events)
                print(
                    f"[dsa-a2a-time rank {rank}] {name}: calls={len(events)} "
                    f"total_ms={total:.1f} mean_ms={total / max(1, len(events)):.3f}",
                    flush=True,
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
        key_positions: The inverse of ``remap`` -- the global position of each received row,
            in row order. The kernels used to read causality out of this, indexed by the
            gathered row number; they now take the global ``topk_indices`` instead, which is
            the same information laid out parallel to the gather indices and so reads as a
            coalesced load rather than a gather depending on a value just loaded. Kept because
            it is the plan's statement of what each received row is.
        total_recv: Number of rows this rank receives, i.e. the size of the exchanged buffer.
        send_split: ``send_counts`` as a host-side list.
        recv_split: ``recv_counts`` as a host-side list.

    ``all_to_all_single`` takes its split sizes as Python lists, so they have to come off the
    device at some point, and every such read stalls the pipeline until the queued work drains.
    They are pulled once here and carried on the plan because the alternative -- calling
    ``.tolist()`` at each use -- repeats that stall for every exchange and every gradient
    return, several times per tile per layer, for numbers that never change.
    """

    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    requested_local: torch.Tensor
    remap: torch.Tensor
    key_positions: torch.Tensor
    total_recv: int
    send_split: list
    recv_split: list


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
    plan: Optional[TileExchangePlan] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[TileExchangePlan]]:
    """Fetch a tile's selected keys and values, or pass through when there is no exchange.

    Returns ``(key, value, gather_indices, plan)``. ``gather_indices`` addresses the returned
    tensors and is None when they are the originals; callers keep ``topk_indices`` for causal
    masking either way. The plan comes back so a backward can hand gradients on the returned
    buffers to :func:`return_row_grads`.

    Pass ``plan`` to reuse one the forward already built. Planning costs a collective and a
    device read, and the recomputing backward would otherwise redo both for a tile whose top-k
    it has just reproduced bit for bit -- the plan is a pure function of those indices, so the
    second computation can only agree with the first.

    Runs without autograd. The min-memory backward recomputes tiles and routes dK and dV by
    hand, so a graph here would be dead weight; see :func:`exchange_rows`.
    """
    if context is None:
        return key, value, None, None
    if plan is None:
        with _TimedRegion("plan"):
            plan = plan_tile_exchange(
                topk_indices, context.seq_len, context.cp_size, context.cp_group
            )
    with _TimedRegion("exchange_kv"):
        tile_key, tile_value = exchange_rows_pair(key, value, plan, context.cp_group)
    with _TimedRegion("remap_gather"):
        gather_indices = plan.remap[topk_indices]
    return tile_key, tile_value, gather_indices, plan


_POSITION_TABLE_CACHE: dict = {}


def _zigzag_position_table(seq_len: int, cp_size: int, device) -> torch.Tensor:
    """``table[r, j]`` is the global position of rank ``r``'s local row ``j``.

    The forward direction of :func:`zigzag_cp_owner_and_local_index`, for every rank at once.
    Reading a mask over the sequence through this gives the same mask laid out per owner,
    which is what the planner needs and what lets it avoid the owner arithmetic entirely. It
    depends only on the layout, so it is built once per shape rather than once per tile.
    """
    key = (seq_len, cp_size, str(device))
    table = _POSITION_TABLE_CACHE.get(key)
    if table is None:
        chunk_len = seq_len // (2 * cp_size)
        rank = torch.arange(cp_size, device=device).unsqueeze(1)
        local = torch.arange(seq_len // cp_size, device=device).unsqueeze(0)
        # Rank r holds chunk r in its low rows and chunk 2*cp_size-1-r in its high rows.
        table = torch.where(
            local < chunk_len,
            rank * chunk_len + local,
            (2 * cp_size - 1 - rank) * chunk_len + (local - chunk_len),
        )
        _POSITION_TABLE_CACHE[key] = table
    return table


def plan_tile_exchange(
    topk_indices: torch.Tensor, seq_len: int, cp_size: int, cp_group
) -> TileExchangePlan:
    """Work out who holds this tile's selected keys and arrange the swap.

    Deduplicates across everything it is given -- every batch row and every query -- because a
    key selected by many queries only has to cross the wire once. Pass a list of tiles to plan
    one exchange for all of them: the dedupe then spans the tiles too, and since neighbouring
    query tiles select heavily overlapping keys, one exchange for a whole layer moves far less
    than one exchange per tile and costs a single collective instead of one each.

    Args:
        topk_indices: Global key positions selected, ``[batch, query, topk]``, or a list of
            such tensors to plan a single exchange covering all of them.
        seq_len: Global sequence length.
        cp_size: Context-parallel world size.
        cp_group: The context-parallel process group.

    Returns:
        The plan the exchange and the index remapping both read.
    """
    selections = topk_indices if isinstance(topk_indices, (list, tuple)) else (topk_indices,)
    device = selections[0].device
    shard_len = seq_len // cp_size

    # Everything before the collective is kept free of device reads, deliberately.
    #
    # A read stalls the host until the queued work drains, and a stall before a collective is
    # paid twice: once draining, then again at the collective, which waits for the slowest
    # rank regardless. Profiling the two-collective version showed exactly that -- the
    # ``nonzero`` that compacted the selections cost about as much as all the arithmetic
    # around it, while the identical reads placed after the collective were almost free,
    # because by then the queue had already drained. So the request is built with dense
    # arithmetic whose shapes are known in advance, and the compaction waits its turn.
    with _TimedRegion("plan.build_request"):
        # Dedupe into a mask over the sequence, then read that mask through the layout table to
        # get the request laid out per owner. No ``nonzero`` and no owner arithmetic: both
        # produce or consume a variable-length list, and the length is a device value, so
        # either one stalls the host here. A gather through a cached table is the same answer
        # at a fixed shape.
        seen = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for selection in selections:
            seen[selection.reshape(-1)] = True
        table = _zigzag_position_table(seq_len, cp_size, device)
        request = seen[table]

    # Ask for rows as a fixed-size mask over each owner's shard rather than as a list of
    # indices. A variable-length request has to be preceded by its own length, which is why
    # this used to be two collectives: counts first, then the indices themselves. A mask is
    # the same size no matter how much it asks for, so no length has to be agreed in advance,
    # and both sides can derive their split sizes from what they already hold -- the requester
    # from its own mask, the owner by counting bits in what arrived.
    #
    # It is also smaller. The mask is one bit per row of a shard, ``cp_size * shard_len`` bits
    # in total, against eight bytes for every index a tile selected.
    with _TimedRegion("plan.request_a2a"):
        incoming = torch.empty_like(request)
        torch.distributed.all_to_all_single(incoming, request, group=cp_group)

    with _TimedRegion("plan.counts"):
        send_counts = request.sum(dim=1).to(torch.int64)
        recv_counts = incoming.sum(dim=1).to(torch.int64)

    # The one device read in the whole planner.
    #
    # Three were needed before: two ``nonzero`` compactions, each of which must learn its own
    # output length, and this one for the split sizes. Each stalls the host until the queue
    # drains, and the profile showed the exchange's cost staying flat across a run while the
    # all-gather arm warmed up -- the signature of a fixed per-iteration stall rather than of
    # work. Both compactions are rewritten below to take their lengths from this read instead.
    with _TimedRegion("plan.splits_sync"):
        both_splits = torch.stack((send_counts, recv_counts)).tolist()
        send_split, recv_split = both_splits[0], both_splits[1]
        total_send, total_recv_rows = sum(send_split), sum(recv_split)

    with _TimedRegion("plan.compact"):
        # Compaction by sorting a sentinel-padded index instead of by ``nonzero``: the wanted
        # entries carry their own index and the rest carry a value that sorts past the end, so
        # the first n of the result are exactly what nonzero would have returned -- but the
        # shapes are fixed, so nothing has to be read back to produce it. n comes from the
        # read above. The arrays sorted are one entry per row of the sequence, not per
        # selection, so this is a few thousand elements.
        cells = request.numel()
        positions = torch.arange(cells, device=device)
        # Ascending flat index over [cp_size, shard_len] is rank-major with local index
        # ascending, which is the order the data collective splits on and the order the
        # replies arrive in; ascending local index is ascending global position within an
        # owner, so this is also what the remap expects.
        chosen = torch.where(request.reshape(-1), positions, cells).sort().values[:total_send]
        wanted_by_owner = table.reshape(-1)[chosen]
        served = torch.where(incoming.reshape(-1), positions, cells).sort().values
        requested_local = served[:total_recv_rows] % shard_len

    # Replies arrive in the order the requests left, so row i of the received buffer is
    # wanted_by_owner[i]. Inverting that gives the remap the gather needs.
    with _TimedRegion("plan.remap_build"):
        remap = torch.zeros(seq_len, dtype=torch.int64, device=device)
        remap[wanted_by_owner] = torch.arange(
            wanted_by_owner.numel(), dtype=torch.int64, device=device
        )

    if _STATS_ENABLED:
        _stats["tiles"] += 1
        _stats["rows_received"] += int(wanted_by_owner.numel())
        # What an all-gather would have moved to this rank for the same tile: everyone else's
        # shard. The ratio of the two is the whole case for the sparse exchange.
        _stats["rows_if_all_gathered"] += seq_len - seq_len // cp_size

    return TileExchangePlan(
        send_counts=send_counts,
        recv_counts=recv_counts,
        requested_local=requested_local,
        remap=remap,
        # Row i of the received buffer holds this global position; the exact inverse of remap.
        key_positions=wanted_by_owner,
        total_recv=int(wanted_by_owner.numel()),
        send_split=send_split,
        recv_split=recv_split,
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
        output_split_sizes=plan.send_split,
        input_split_sizes=plan.recv_split,
        group=cp_group,
    )
    return received


def exchange_rows_pair(
    key_rows: torch.Tensor, value_rows: torch.Tensor, plan: TileExchangePlan, cp_group
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exchange K and V in a single collective rather than one each.

    K and V are selected by the same top-k, so they address the same rows and move between the
    same pairs of ranks. Sending them as one buffer halves the number of collectives, which is
    what this costs: the rows are the handful the top-k picked, so the payload is small enough
    that per-call launch and handshake latency, not bandwidth, sets the price.

    They are joined on the trailing dimension because the split sizes partition dimension 0 --
    anything concatenated there would have to interleave with the rank blocks. The halves come
    back as views, so each is made contiguous for the kernels, a copy of the received rows only.
    """
    head_dim = key_rows.shape[-1]
    served = torch.cat(
        (
            key_rows.index_select(0, plan.requested_local),
            value_rows.index_select(0, plan.requested_local),
        ),
        dim=-1,
    )
    received = torch.empty(
        (plan.total_recv, *served.shape[1:]), dtype=served.dtype, device=served.device
    )
    torch.distributed.all_to_all_single(
        received,
        served.contiguous(),
        output_split_sizes=plan.send_split,
        input_split_sizes=plan.recv_split,
        group=cp_group,
    )
    return received[..., :head_dim].contiguous(), received[..., head_dim:].contiguous()


def return_row_grads_pair(
    grad_key_received: torch.Tensor,
    grad_value_received: torch.Tensor,
    plan: TileExchangePlan,
    cp_group,
    local_shape,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route dK and dV home together, the mirror of :func:`exchange_rows_pair`.

    Same reasoning: one collective instead of two for payloads small enough that the call
    itself dominates.
    """
    timer = _TimedRegion("return_grads")
    timer.__enter__()
    head_dim = grad_key_received.shape[-1]
    grad_received = torch.cat((grad_key_received, grad_value_received), dim=-1)
    grad_served = torch.empty(
        (plan.requested_local.numel(), *grad_received.shape[1:]),
        dtype=grad_received.dtype,
        device=grad_received.device,
    )
    torch.distributed.all_to_all_single(
        grad_served,
        grad_received.contiguous(),
        output_split_sizes=plan.recv_split,
        input_split_sizes=plan.send_split,
        group=cp_group,
    )
    grad_local = torch.zeros(
        (*local_shape[:-1], 2 * head_dim), dtype=grad_received.dtype, device=grad_received.device
    )
    grad_local.index_add_(0, plan.requested_local, grad_served)
    split = grad_local[..., :head_dim].contiguous(), grad_local[..., head_dim:].contiguous()
    timer.__exit__()
    return split


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
        output_split_sizes=plan.recv_split,
        input_split_sizes=plan.send_split,
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
