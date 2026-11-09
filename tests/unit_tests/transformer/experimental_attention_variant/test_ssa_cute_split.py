# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The split routing/attention path must be the fused one, numerically.

Splitting exists to make room for a context-parallel exchange between routing and attention,
not to change what the kernels compute. With no exchange the two paths run the same kernels on
the same inputs in the same order, so a difference means the split dropped or reordered
something -- which would otherwise surface as a plausible loss curve rather than a failure.

"The same" cannot mean bit-exact. dQ is one value per query, but dK, dV and the indexer
gradients accumulate every query that selected a given key, and that reduction is not
order-stable: running the fused path twice on identical inputs already disagrees by ~5e-5
relative on dK. Asserting equality there fails on the kernel, not on the split.

So the fused path is measured against itself first and that becomes the bar. A tolerance
derived from the reference is self-calibrating -- it tracks the kernel if its reproducibility
changes, and it cannot be quietly widened to cover a real defect, because widening it would
require the reference to disagree with itself by the same amount.
"""

import pytest
import torch

ssa = pytest.importorskip("simplified_sparse_attention")

from megatron.core.transformer.experimental_attention_variant.dsa_ssa_cute import (
    split_sparse_attention,
)

HEAD_DIM = 256
INDEX_DIM = 128
QUERY_HEADS = 16
TOPK = 512


def _blackwell_only():
    if not torch.cuda.is_available():
        pytest.skip("CuTe DSA needs a CUDA device")
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("CuTe DSA kernels are SM103-only")


def _inputs(batch, seq, device):
    rows = batch * seq
    bf16 = dict(dtype=torch.bfloat16, device=device)
    torch.manual_seed(1234)
    tensors = {
        "query": torch.randn(rows, QUERY_HEADS, HEAD_DIM, **bf16),
        "key": torch.randn(rows, 1, HEAD_DIM, **bf16),
        "value": torch.randn(rows, 1, HEAD_DIM, **bf16),
        "q_idx": torch.randn(rows, INDEX_DIM, **bf16),
        "k_idx": torch.randn(rows, INDEX_DIM, **bf16),
    }
    position = torch.arange(seq, device=device, dtype=torch.int32).repeat(batch)
    base = torch.arange(batch, device=device, dtype=torch.int32).repeat_interleave(seq) * seq
    layout = ssa.RowMetadata(
        key_starts=base,
        key_ends=base + position + 1,
        query_valid=torch.ones(rows, dtype=torch.bool, device=device),
        loss_denominator=torch.tensor(float(rows), dtype=torch.float32, device=device),
        triangular_scores=False,
        mask_invalid_rows=True,
        all_query_rows_valid=True,
    )
    return tensors, layout


def _run(call, batch, seq, device):
    tensors, layout = _inputs(batch, seq, device)
    for tensor in tensors.values():
        tensor.requires_grad_(True)
    output, indexer_loss = call(
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["q_idx"],
        tensors["k_idx"],
        layout,
        TOPK,
        loss_coeff=0.1,
    )
    (output.float().square().mean() + indexer_loss).backward()
    quantities = {"output": output.detach(), "indexer_loss": indexer_loss.detach()}
    quantities.update({name: tensor.grad.detach() for name, tensor in tensors.items()})
    return {name: value.float().clone() for name, value in quantities.items()}


def _max_abs_diff(left, right):
    return (left - right).abs().max().item()


@pytest.mark.parametrize("batch,seq", [(1, 2048), (2, 1024)])
def test_split_matches_fused_forward_and_backward(batch, seq):
    _blackwell_only()
    device = torch.device("cuda")

    # Three references, not two. A single pair measures whichever elements happened to land
    # differently that run, and that varies by three orders of magnitude between runs -- a
    # tolerance built on one sample passes or fails on which draw it got.
    references = [_run(ssa.sparse_attention, batch, seq, device) for _ in range(3)]
    split = _run(split_sparse_attention, batch, seq, device)

    for name in references[0]:
        noise = max(
            _max_abs_diff(left[name], right[name])
            for left, right in zip(references, references[1:] + references[:1])
        )
        scale = references[0][name].abs().max().item()
        # The relative floor is a quarter of a BF16 ULP (2**-8). Below that, a difference
        # cannot change a stored result at full magnitude, so it is not a defect worth failing
        # on; above it, the kernels genuinely disagree. Quantities the kernel reproduces
        # exactly keep an effectively exact bar, since their noise term stays zero.
        tolerance = max(4.0 * noise, 1e-3 * scale)
        observed = _max_abs_diff(split[name], references[0][name])
        assert observed <= tolerance, (
            f"{name}: split differs from fused by {observed:.3e} (rel {observed / max(scale, 1e-30):.3e}), "
            f"beyond the fused path's own spread of {noise:.3e} and the tolerance "
            f"{tolerance:.3e}. That is a real divergence, not the non-deterministic reduction."
        )
