"""Smallest end-to-end exercise of the simplified-sparse-attention CuTe kernels.

Proves the install, the SM103 gate, the shape envelope and the RowMetadata convention that
dsa_gqa._forward_ssa_cute builds, without a training job. Run inside the container on a GPU:

    python ssa_smoke.py
"""

import torch
from simplified_sparse_attention import RowMetadata, reserve_indexer, sparse_attention

HEAD_DIM = 256
INDEX_DIM = 128
QUERY_HEADS = 16  # supported: 16, 32, 96
TOPK = 512  # supported: 512, 1024, 2048
BATCH = 2
SEQ = 2048


def main() -> None:
    device = torch.device("cuda")
    print("capability:", torch.cuda.get_device_capability(device))

    rows = BATCH * SEQ
    bf16 = dict(dtype=torch.bfloat16, device=device)
    query = torch.randn(rows, QUERY_HEADS, HEAD_DIM, **bf16).requires_grad_(True)
    key = torch.randn(rows, 1, HEAD_DIM, **bf16).requires_grad_(True)
    value = torch.randn(rows, 1, HEAD_DIM, **bf16).requires_grad_(True)
    index_q = torch.randn(rows, INDEX_DIM, **bf16).requires_grad_(True)
    index_k = torch.randn(rows, INDEX_DIM, **bf16).requires_grad_(True)

    # Same layout dsa_gqa._forward_ssa_cute builds: batch-major rows, each sequence one
    # document, each row's causal interval given explicitly.
    position = torch.arange(SEQ, device=device, dtype=torch.int32).repeat(BATCH)
    base = torch.arange(BATCH, device=device, dtype=torch.int32).repeat_interleave(SEQ) * SEQ
    layout = RowMetadata(
        key_starts=base,
        key_ends=base + position + 1,
        query_valid=torch.ones(rows, dtype=torch.bool, device=device),
        loss_denominator=torch.tensor(float(rows), dtype=torch.float32, device=device),
        triangular_scores=False,
        mask_invalid_rows=True,
        all_query_rows_valid=True,
    )

    reserve_indexer(rows, rows, TOPK)
    output, indexer_loss = sparse_attention(
        query, key, value, index_q, index_k, layout, TOPK, loss_coeff=0.1
    )
    print("output:", tuple(output.shape), output.dtype)
    print("indexer_loss:", float(indexer_loss))
    assert output.shape == query.shape, (output.shape, query.shape)
    assert torch.isfinite(indexer_loss).all(), "indexer loss is not finite"

    (output.float().square().mean() + indexer_loss).backward()
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("value", value),
        ("index_q", index_q),
        ("index_k", index_k),
    ):
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is not finite"
    print("backward OK")


if __name__ == "__main__":
    main()
