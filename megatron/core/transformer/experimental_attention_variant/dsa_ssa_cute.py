# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Routing and attention as two steps in the simplified-sparse-attention CuTe path.

The wheel's public ``sparse_attention`` fuses indexer top-k and attention into one autograd
call, which leaves nowhere to put a context-parallel key exchange: a rank cannot know which
keys to fetch until routing has chosen them, and by then the selection never leaves the kernel.
This reproduces that call against the same private entry points with the seam left open, so an
exchange can sit between the two halves.

The seam is real rather than invented -- ``backend._run_sparse_forward`` is itself two kernel
launches with a materialised ``[Tq, topk]`` index tensor between them, and the wheel already
defers its main-K/V join until after routing for exactly this reason. What is missing upstream
is a public entry point, not the structure.

Numerically this is the public path. With no exchange it must agree bit-for-bit, which is what
``tests/.../test_ssa_cute_split.py`` pins: the point of the split is where the collective goes,
not what the kernels compute.
"""

from typing import Optional

import torch

# Private wheel entry points. They are plain Python, so this needs no change to the wheel, but
# it does bind to names the package does not promise: pinned to 0.1.0.dev14 and checked at
# import so a version bump fails loudly here instead of somewhere numerical.
try:
    from simplified_sparse_attention.backend import (
        _attention_forward_entry,
        _kernel_module,
        _normalize_layout,
        _validate_inputs,
    )

    HAVE_SSA_CUTE_INTERNALS = True
except ImportError:  # pragma: no cover - exercised only where the wheel is absent
    HAVE_SSA_CUTE_INTERNALS = False


def require_ssa_cute_internals() -> None:
    """Fail with the reason rather than an AttributeError deep in a forward."""
    if not HAVE_SSA_CUTE_INTERNALS:
        raise RuntimeError(
            "The split ssa-cute path needs simplified_sparse_attention's private entry points "
            "(_kernel_module, _attention_forward_entry, _normalize_layout). Install the "
            "0.1.0.dev14 wheel; a later version may have renamed them."
        )


class SplitSparseAttention(torch.autograd.Function):
    """``sparse_attention`` with routing and attention separated.

    ``exchange`` is called once with the globally-indexed selection and returns the key/value
    buffers attention should read plus the selection rewritten into that buffer's coordinates.
    Returning the inputs unchanged reproduces the fused path.
    """

    @staticmethod
    def forward(ctx, query, key, value, q_idx, k_idx, layout, topk, loss_coeff, exchange):
        require_ssa_cute_internals()
        _validate_inputs(query, key, value, q_idx, k_idx)
        layout, lengths, attention_lengths = _normalize_layout(
            layout,
            query_length=query.size(0),
            key_length=key.size(0),
            device=query.device,
            topk=topk,
        )

        rows = query.size(0)
        # Allocated up front, in the order the fused path allocates them, and deliberately so.
        # indexer_aux_grad_precompute reads every one of a row's topk teacher slots, including
        # the ones past its valid length that attention never writes. Those slots hold whatever
        # the caching allocator last left at that address, so the allocation sequence is part of
        # the numerics: interleaving these with the routing call changes the indexer gradient in
        # a handful of elements, which is small enough to read as noise and is not.
        indices = torch.empty((rows, topk), device=query.device, dtype=torch.int32)
        predict = torch.empty_like(indices, dtype=torch.float32)
        output = torch.empty_like(query)
        lse = torch.empty((rows, query.size(1)), device=query.device, dtype=torch.float32)
        teacher = torch.empty_like(predict)
        softmax_max = torch.empty_like(lse)
        softmax_inv_sum = torch.empty_like(lse)

        # --- routing: selects over the key stream k_idx describes, in its coordinates ---
        _kernel_module("indexer_forward_topk").run(
            q_idx,
            k_idx,
            indices,
            predict,
            layout.key_starts,
            layout.key_ends,
            triangular=layout.triangular_scores,
            score_impl="auto",
            fusion="auto",
            topk_impl="auto",
            topk_overflow_policy="REREAD_ALWAYS",
        )
        if layout.mask_invalid_rows:
            # Padding rows route nowhere; a one-slot sentinel keeps every gather in range even
            # if a top-k implementation leaves an empty row unwritten.
            indices.masked_fill_(~layout.query_valid[:, None], 0)

        # --- the seam: everything above is global, everything below reads one buffer ---
        attention_key, attention_value, attention_indices, exchange_state = exchange(
            key, value, indices
        )

        _attention_forward_entry(query_heads=query.size(1), topk=topk)(
            query,
            attention_key,
            attention_value,
            attention_indices,
            attention_lengths,
            output,
            lse,
            teacher,
            softmax_max,
            softmax_inv_sum,
        )
        if layout.mask_invalid_rows:
            output.masked_fill_(~layout.query_valid.view(-1, 1, 1), 0.0)

        # --- indexer objective: global coordinates again, since it scores against k_idx ---
        indexer_loss = torch.empty((), device=query.device, dtype=torch.float32)
        unit_dq_idx = torch.empty_like(q_idx)
        unit_dk_idx = torch.empty_like(k_idx)
        # defer_dk is deliberately not used. Upstream enables it only for one exact shape
        # (8192 queries, 512K/1M keys, top-k 2048); a context-parallel split never matches it,
        # and its deferred edge records would have to survive the exchange bookkeeping too.
        _kernel_module("indexer_aux_grad_precompute").run(
            q_idx,
            k_idx,
            indices,
            lengths,
            predict,
            teacher,
            indexer_loss,
            unit_dq_idx,
            unit_dk_idx,
            layout.loss_denominator,
            loss_coeff=loss_coeff,
            rectangular=not layout.triangular_scores,
            spmm_impl="auto",
        )
        if not layout.all_query_rows_valid:
            unit_dq_idx.masked_fill_(~layout.query_valid.view(-1, 1), 0.0)

        ctx.save_for_backward(
            query,
            attention_key,
            attention_value,
            output,
            lse,
            attention_indices,
            attention_lengths,
            layout.query_valid,
            unit_dq_idx,
            unit_dk_idx,
        )
        ctx.all_query_rows_valid = layout.all_query_rows_valid
        ctx.fuse_boundary_ops = layout.fuse_boundary_ops
        ctx.exchange_state = exchange_state
        ctx.set_materialize_grads(False)
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_indexer_loss):
        (
            query,
            attention_key,
            attention_value,
            output,
            lse,
            attention_indices,
            attention_lengths,
            query_valid,
            unit_dq_idx,
            unit_dk_idx,
        ) = ctx.saved_tensors

        grad_query = grad_key = grad_value = None
        if grad_output is not None and any(ctx.needs_input_grad[:3]):
            grad_output = grad_output.contiguous().view_as(query)
            if not ctx.all_query_rows_valid:
                # contiguous() may hand back caller-owned storage; never mask it in place.
                grad_output = grad_output.masked_fill(~query_valid.view(-1, 1, 1), 0.0)
            dq = torch.empty(query.shape, device=query.device, dtype=query.dtype)
            dk = torch.empty(attention_key.shape, device=query.device, dtype=attention_key.dtype)
            dv = torch.empty(
                attention_value.shape, device=query.device, dtype=attention_value.dtype
            )
            backward = _kernel_module("sparse_attention_backward", query_heads=query.size(1))
            if ctx.fuse_boundary_ops and query.size(1) == 96:
                backward = _kernel_module("sparse_attention_workspace")
            backward.run(
                query,
                attention_key,
                attention_value,
                output,
                grad_output,
                lse,
                attention_indices,
                attention_lengths,
                dq,
                dk,
                dv,
            )
            grad_query = dq if ctx.needs_input_grad[0] else None
            # dk/dv are in the attention buffer's coordinates. Whoever supplied that buffer
            # owns putting them back; with no exchange the buffer is the input.
            grad_key, grad_value = ctx.exchange_state.return_grads(dk, dv)

        grad_q_idx = grad_k_idx = None
        if grad_indexer_loss is not None:
            if ctx.needs_input_grad[3]:
                grad_q_idx = unit_dq_idx * grad_indexer_loss.to(unit_dq_idx.dtype)
            if ctx.needs_input_grad[4]:
                grad_k_idx = unit_dk_idx * grad_indexer_loss.to(unit_dk_idx.dtype)
        return (grad_query, grad_key, grad_value, grad_q_idx, grad_k_idx, None, None, None, None)


class _NoExchange:
    """Attention reads the keys it was given; gradients need no return trip."""

    def __call__(self, key, value, indices):
        return key, value, indices, self

    @staticmethod
    def return_grads(grad_key, grad_value):
        return grad_key, grad_value


class SelectedKVExchange:
    """Fetch only the keys routing selected, from the ranks that own them.

    Reuses the min-memory path's planner and collectives unchanged. Two coordinate systems meet
    here and the translation between them is the whole job:

    * the plan addresses ``[local_seq, batch, groups, head_dim]`` and dedupes over *sequence
      positions*, spanning the batch, because a key selected by many queries crosses the wire
      once;
    * the CuTe kernels address a flat batch-major token stream, where row ``b * seq + p`` is
      batch ``b``'s position ``p``.

    So selections arrive as flat stream indices, are reduced to positions for the plan, and the
    plan's answer is lifted back into flat rows of the received buffer.
    """

    def __init__(self, cp_group, cp_size: int, global_seq: int, batch_size: int, local_seq: int):
        self.cp_group = cp_group
        self.cp_size = cp_size
        self.global_seq = global_seq
        self.batch_size = batch_size
        self.local_seq = local_seq
        self._plan = None
        self._head_dim = None

    def __call__(self, key, value, indices):
        from megatron.core.transformer.experimental_attention_variant.dsa_a2a import (
            exchange_rows_pair,
            plan_tile_exchange,
        )

        batch, gseq = self.batch_size, self.global_seq
        head_dim = key.size(-1)
        # Slots past a row's valid length hold whatever the top-k kernel left there. They are
        # never attended to, but they are still gathered through remap, so bring them in range.
        flat = indices.long().clamp_(0, batch * gseq - 1)
        batch_id, position = flat // gseq, flat % gseq

        plan = plan_tile_exchange(
            position.view(batch, -1, position.size(-1)), gseq, self.cp_size, self.cp_group
        )
        # [b*s, 1, D] -> [s, b, 1, D]: the plan selects rows of dimension 0, which it reads as
        # sequence position with the batch alongside.
        key_sb = key.view(batch, self.local_seq, 1, head_dim).transpose(0, 1).contiguous()
        value_sb = value.view(batch, self.local_seq, 1, head_dim).transpose(0, 1).contiguous()
        received_key, received_value = exchange_rows_pair(key_sb, value_sb, plan, self.cp_group)

        total = plan.total_recv
        flat_key = received_key.transpose(0, 1).reshape(batch * total, 1, head_dim).contiguous()
        flat_value = received_value.transpose(0, 1).reshape(batch * total, 1, head_dim).contiguous()
        # The received buffer is per-batch too, so a flat row is batch-offset the same way the
        # global stream is -- with the received count in place of the sequence length.
        compact = (batch_id * total + plan.remap[position]).to(torch.int32)

        self._plan, self._head_dim = plan, head_dim
        return flat_key, flat_value, compact, self

    def return_grads(self, grad_key, grad_value):
        from megatron.core.transformer.experimental_attention_variant.dsa_a2a import (
            return_row_grads_pair,
        )

        plan, head_dim = self._plan, self._head_dim
        batch, total = self.batch_size, plan.total_recv
        grad_key_sb = grad_key.view(batch, total, 1, head_dim).transpose(0, 1).contiguous()
        grad_value_sb = grad_value.view(batch, total, 1, head_dim).transpose(0, 1).contiguous()
        local_key, local_value = return_row_grads_pair(
            grad_key_sb, grad_value_sb, plan, self.cp_group, (self.local_seq, batch, 1, head_dim)
        )
        return (
            local_key.transpose(0, 1).reshape(batch * self.local_seq, 1, head_dim).contiguous(),
            local_value.transpose(0, 1).reshape(batch * self.local_seq, 1, head_dim).contiguous(),
        )


def split_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    layout,
    topk: int,
    *,
    loss_coeff: float = 0.1,
    exchange: Optional[object] = None,
):
    """Public shape of :class:`SplitSparseAttention`; see the module docstring.

    Args:
        exchange: Called as ``exchange(key, value, indices)`` after routing, returning
            ``(key, value, indices, state)`` where ``state.return_grads(dk, dv)`` maps the
            attention buffer's gradients back onto ``key``/``value``. ``None`` reproduces the
            fused path exactly.
    """
    return SplitSparseAttention.apply(
        query, key, value, q_idx, k_idx, layout, topk, float(loss_coeff), exchange or _NoExchange()
    )
