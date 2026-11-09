"""Is the split path's dK wrong, or is dK just not reproducible?

dQ is one value per query; dK and dV accumulate every query that selected the same key, so if
the kernel reduces with atomics their summation order varies between runs. That produces the
same symptom as a real bug at rtol=0.

So measure the fused path against *itself* first. If fused-vs-fused differs by about as much as
split-vs-fused, the kernel is the source and the test's tolerance was wrong. If fused-vs-fused
is exactly zero and split-vs-fused is not, the split dropped something.

    python .bench_logs/ssa_ab/diag_split_dk.py
"""

import torch
from simplified_sparse_attention import RowMetadata, sparse_attention

from megatron.core.transformer.experimental_attention_variant.dsa_ssa_cute import (
    split_sparse_attention,
)

HEAD_DIM, INDEX_DIM, QUERY_HEADS, TOPK = 256, 128, 16, 512
BATCH, SEQ = 1, 2048


def build(device):
    rows = BATCH * SEQ
    torch.manual_seed(1234)
    bf16 = dict(dtype=torch.bfloat16, device=device)
    tensors = {
        "query": torch.randn(rows, QUERY_HEADS, HEAD_DIM, **bf16),
        "key": torch.randn(rows, 1, HEAD_DIM, **bf16),
        "value": torch.randn(rows, 1, HEAD_DIM, **bf16),
        "q_idx": torch.randn(rows, INDEX_DIM, **bf16),
        "k_idx": torch.randn(rows, INDEX_DIM, **bf16),
    }
    pos = torch.arange(SEQ, device=device, dtype=torch.int32).repeat(BATCH)
    base = torch.arange(BATCH, device=device, dtype=torch.int32).repeat_interleave(SEQ) * SEQ
    layout = RowMetadata(
        key_starts=base,
        key_ends=base + pos + 1,
        query_valid=torch.ones(rows, dtype=torch.bool, device=device),
        loss_denominator=torch.tensor(float(rows), dtype=torch.float32, device=device),
        triangular_scores=False,
        mask_invalid_rows=True,
        all_query_rows_valid=True,
    )
    return tensors, layout


def run(call, device):
    tensors, layout = build(device)
    for t in tensors.values():
        t.requires_grad_(True)
    out, loss = call(
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["q_idx"],
        tensors["k_idx"],
        layout,
        TOPK,
        loss_coeff=0.1,
    )
    (out.float().square().mean() + loss).backward()
    return {k: v.grad.detach().float().clone() for k, v in tensors.items()}


def report(title, a, b):
    print(f"\n{title}")
    for name in a:
        d = (a[name] - b[name]).abs()
        scale = a[name].abs().max().clamp_min(1e-30)
        print(
            f"  {name:<7} max|diff|={d.max().item():.3e}  "
            f"rel={d.max().item() / scale.item():.3e}  "
            f"mismatched={(d > 0).float().mean().item() * 100:.2f}%"
        )


def main():
    device = torch.device("cuda")
    fused_a = run(sparse_attention, device)
    fused_b = run(sparse_attention, device)
    split = run(split_sparse_attention, device)
    report("fused vs fused  (kernel's own reproducibility)", fused_a, fused_b)
    report("split vs fused  (what the test compared)", split, fused_a)
    print(
        "\nIf the two blocks are the same order of magnitude, the kernel is not reproducible "
        "and rtol=0 was the wrong assertion.\nIf fused-vs-fused is all zeros and split-vs-fused "
        "is not, the split is wrong."
    )


if __name__ == "__main__":
    main()
