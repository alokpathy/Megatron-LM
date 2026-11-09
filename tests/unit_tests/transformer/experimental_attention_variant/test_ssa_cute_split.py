# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The split routing/attention path must be the fused one, numerically.

Splitting exists to make room for a context-parallel exchange between routing and attention,
not to change what the kernels compute. With no exchange the two paths run the same kernels on
the same inputs in the same order, so anything but bit-exact agreement means the split dropped
or reordered something -- which would otherwise surface as a plausible loss curve.
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


@pytest.mark.parametrize("batch,seq", [(1, 2048), (2, 1024)])
def test_split_matches_fused_forward_and_backward(batch, seq):
    _blackwell_only()
    device = torch.device("cuda")

    results = []
    for call in (ssa.sparse_attention, split_sparse_attention):
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
        results.append(
            (
                output.detach().clone(),
                indexer_loss.detach().clone(),
                {name: tensor.grad.detach().clone() for name, tensor in tensors.items()},
            )
        )

    fused, split = results
    torch.testing.assert_close(split[0], fused[0], rtol=0, atol=0)
    torch.testing.assert_close(split[1], fused[1], rtol=0, atol=0)
    for name in fused[2]:
        torch.testing.assert_close(split[2][name], fused[2][name], rtol=0, atol=0, msg=name)
