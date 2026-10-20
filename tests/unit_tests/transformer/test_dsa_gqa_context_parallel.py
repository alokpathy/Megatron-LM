# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the DSA-over-GQA context-parallel helpers.

Covers the pieces that need no process group: the scalar that carries a tile's global position,
the tiling validation, and the gather helper's guards and zigzag reorder. These run on any GPU
count, which matters because the real cp=2 configuration needs ranks this cluster allocates in
groups of four alongside TP.

End-to-end parity -- output, gradients, and the collectives themselves -- lives in
test_dsa_gqa_context_parallel_dist.py, which runs under torch.distributed against a live group.
Emulating a post-gather rank single-process is not possible for the learned-K router: its keys
come from hidden_states, so a local hidden slice yields an index covering only this rank.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    _gather_kv_for_context_parallel,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_zigzag_allgather_cp_key_reorder,
    build_zigzag_cp_local_positions,
    zigzag_cp_owner_and_local_index,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    _tile_global_start,
    _validate_query_positions,
)

SEQLEN, BATCH, HIDDEN = 16, 2, 12
HEADS, HEAD_DIM = 2, 4
CP_SIZE = 2


def _inputs(seed=0, requires_grad=False):
    torch.manual_seed(seed)
    q = torch.randn(SEQLEN, BATCH, HEADS, HEAD_DIM, requires_grad=requires_grad)
    k = torch.randn(SEQLEN, BATCH, 1, HEAD_DIM, requires_grad=requires_grad)
    v = torch.randn(SEQLEN, BATCH, 1, HEAD_DIM, requires_grad=requires_grad)
    h = torch.randn(SEQLEN, BATCH, HIDDEN)
    return q, k, v, h


class TestPositionHelpers:
    """The scalar plumbing that carries a tile's global position."""

    def test_tile_global_start_is_identity_without_cp(self):
        assert _tile_global_start(None, 0) == 0
        assert _tile_global_start(None, 7) == 7

    def test_tile_global_start_reads_the_positions_tensor(self):
        pos = build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, 1, torch.device("cpu"))
        # rank 1 of 2 holds chunks 1 and 2 of 4, i.e. global 4..7 then 8..11
        assert pos.tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
        assert _tile_global_start(pos, 0) == 4
        assert _tile_global_start(pos, 4) == 8

    def test_validate_accepts_tiles_that_stay_within_a_chunk(self):
        pos = build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, 0, torch.device("cpu"))
        _validate_query_positions(pos, pos.numel(), 4)
        _validate_query_positions(pos, pos.numel(), 2)

    def test_validate_rejects_a_tile_straddling_two_chunks(self):
        """A tile spanning both zigzag chunks cannot be described by one scalar start."""
        pos = build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, 0, torch.device("cpu"))
        with pytest.raises(ValueError, match="spans a discontinuity"):
            _validate_query_positions(pos, pos.numel(), 8)

    def test_validate_is_a_noop_without_cp(self):
        _validate_query_positions(None, SEQLEN, 8)


class TestZigzagOwnerMap:
    """The inverse direction the all-to-all exchange needs: position -> (owner, local index)."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_round_trips_through_build_zigzag_cp_local_positions(self, cp_size):
        """For every rank and slot, the owner map must land back on the same global position."""
        seq_len = 8 * cp_size
        device = torch.device("cpu")
        for rank in range(cp_size):
            positions = build_zigzag_cp_local_positions(seq_len, cp_size, rank, device)
            owner, local = zigzag_cp_owner_and_local_index(positions, seq_len, cp_size)
            assert torch.equal(owner, torch.full_like(owner, rank))
            assert torch.equal(local, torch.arange(positions.numel()))

    def test_every_global_position_is_owned_exactly_once(self):
        """A dropped or duplicated owner silently loses or double-counts a key."""
        cp_size, seq_len = 4, 32
        positions = torch.arange(seq_len)
        owner, local = zigzag_cp_owner_and_local_index(positions, seq_len, cp_size)
        shard_len = seq_len // cp_size
        seen = torch.zeros(cp_size, shard_len, dtype=torch.bool)
        seen[owner, local] = True
        assert bool(seen.all())

    def test_is_identity_without_cp(self):
        positions = torch.arange(SEQLEN)
        owner, local = zigzag_cp_owner_and_local_index(positions, SEQLEN, 1)
        assert torch.equal(owner, torch.zeros_like(positions))
        assert torch.equal(local, positions)

    def test_rejects_a_sequence_the_layout_cannot_split(self):
        with pytest.raises(ValueError, match="divisible by 2 \\* cp_size"):
            zigzag_cp_owner_and_local_index(torch.arange(6), 6, 4)


class TestGatherHelper:
    """Shape and ordering contract of the all-gather, without a live process group."""

    def test_single_rank_is_a_noop(self):
        q, k, v, _ = _inputs()
        gk, gv, pos, exchange = _gather_kv_for_context_parallel(k, v, None, "allgather")
        assert gk is k and gv is v and pos is None
        # No context parallelism means nothing to exchange either.
        assert exchange is None

    def test_non_allgather_comm_type_is_rejected(self):
        q, k, v, _ = _inputs()
        fake_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
        with pytest.raises(NotImplementedError, match="allgather"):
            _gather_kv_for_context_parallel(k, v, fake_group, "p2p")

    def test_a2a_skips_the_gather_and_returns_an_exchange_context(self):
        """a2a leaves K and V local; the tile loop swaps only what routing selects."""
        q, k, v, _ = _inputs()
        fake_group = SimpleNamespace(size=lambda: CP_SIZE, rank=lambda: 0)
        gk, gv, pos, exchange = _gather_kv_for_context_parallel(k, v, fake_group, "a2a")

        assert gk is k and gv is v
        assert exchange is not None
        # The global length has to be stated, not inferred from the local shard, or routing
        # would clamp causal visibility to one rank's slice.
        assert exchange.seq_len == SEQLEN * CP_SIZE
        assert exchange.cp_size == CP_SIZE
        assert pos is not None and pos.numel() == SEQLEN

    def test_reorder_restores_global_position_order(self):
        """Gathered tensors arrive in rank order; the reorder must sort them by position."""
        device = torch.device("cpu")
        gathered_positions = torch.cat(
            [build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, r, device) for r in range(CP_SIZE)]
        )
        reorder = build_zigzag_allgather_cp_key_reorder(SEQLEN // CP_SIZE, CP_SIZE, device)
        assert gathered_positions[reorder].tolist() == list(range(SEQLEN))
