#!/usr/bin/env python
"""Standalone validation for cuDNN sparse_indexer_score_recompute_wrapper (the DSA
"student" top-k softmax) vs the Megatron reference math. Touches NO training code.

Run inside the training container on a GPU node, e.g.:
    python experiments/validate_cudnn_student.py

It checks that the wrapper's `predict` == softmax over top-k of
    sum_h ReLU(q_h . k_topk[i]) * w_h
including the causal masking of invalid top-k slots (early query positions).

Exit code 0 = match within tolerance; non-zero = mismatch (details printed).
"""
import sys
import torch

try:
    from cudnn import DSA
except Exception:
    from cudnn.deepseek_sparse_attention import DSA


def reference_student(q_index, full_k_index, weights, topk_indices, q_start):
    """Mirror _index_scores_for_selected (:976) + causal mask (:1707) + softmax.

    q_index:      (q_len, B, H, D)
    full_k_index: (S_k,   B, D)      single shared indexer-K head
    weights:      (q_len, B, H)
    topk_indices: (B, q_len, topk)   per-batch-local key positions in [0, S_k)
    returns p:    (B, q_len, topk) FP32
    """
    q_len, B, H, D = q_index.shape
    S_k = full_k_index.shape[0]
    topk = topk_indices.shape[-1]

    q = q_index.permute(1, 0, 2, 3).float()          # (B, q_len, H, D)
    w = weights.permute(1, 0, 2).float()             # (B, q_len, H)
    k_bhd = full_k_index.permute(1, 0, 2).float()    # (B, S_k, D)

    # gather selected K per (b, q): (B, q_len, topk, D)
    idx = topk_indices.long().clamp_(0, S_k - 1)
    sel_k = torch.gather(
        k_bhd.unsqueeze(1).expand(B, q_len, S_k, D),
        2,
        idx.unsqueeze(-1).expand(B, q_len, topk, D),
    )

    scores = torch.einsum("bqhd,bqkd->bqhk", q, sel_k)   # (B, q_len, H, topk)
    scores = torch.relu(scores)
    scores = scores * w.unsqueeze(-1)                    # gate per head
    logits = scores.sum(dim=2)                           # (B, q_len, topk)

    # causal mask: slot invalid if selected key pos > this query's abs position
    qpos = torch.arange(q_start, q_start + q_len, device=idx.device).view(1, q_len, 1)
    invalid = topk_indices > qpos
    logits = logits.masked_fill(invalid, float("-inf"))
    return torch.softmax(logits, dim=-1)                 # (B, q_len, topk)


def build_case(B, q_len, S_k, H, D, topk, q_start, device, dtype, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    q_index = torch.randn(q_len, B, H, D, generator=g, device=device, dtype=dtype)
    full_k = torch.randn(S_k, B, D, generator=g, device=device, dtype=dtype)
    weights = torch.rand(q_len, B, H, generator=g, device=device, dtype=dtype) * 0.05

    # Build causal top-k indices per (b, q): pick <=topk keys with pos <= q_start+q,
    # pad the remainder with (q_start+q+1) so early rows exercise the invalid path,
    # then sort ascending so pads land in the tail (matches _cudnn_indexer_topk_full_k).
    topk_idx = torch.empty(B, q_len, topk, dtype=torch.long, device=device)
    topk_len = torch.empty(B, q_len, dtype=torch.int32, device=device)
    for b in range(B):
        for q in range(q_len):
            qpos = q_start + q
            n_valid = min(topk, qpos + 1)
            valid = torch.randperm(qpos + 1, generator=g, device=device)[:n_valid]
            pad = torch.full((topk - n_valid,), min(qpos + 1, S_k - 1), device=device)
            row = torch.cat([valid, pad]).sort().values
            topk_idx[b, q] = row
            topk_len[b, q] = n_valid
    return q_index, full_k, weights, topk_idx, topk_len


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device"); return 2
    dev = torch.device("cuda")
    # Small but representative of the real config (H=64, D=128); tiny seq for speed.
    B, q_len, S_k, H, D, topk, q_start = 2, 32, 256, 64, 128, 128, 0
    dtype = torch.bfloat16

    q_index, full_k, weights, topk_idx, topk_len = build_case(
        B, q_len, S_k, H, D, topk, q_start, dev, dtype, seed=0
    )

    ref = reference_student(q_index, full_k, weights, topk_idx, q_start)  # (B,q_len,topk)

    # cuDNN layout: q (B,q_len,H,D), k (B,S_k,1,D) shared head, w (B,q_len,H)
    q_bf = q_index.permute(1, 0, 2, 3).contiguous()
    k_bf = full_k.permute(1, 0, 2).contiguous()   # (B, S_k, D) 3-D MQA, no head dim
    w_bf = weights.permute(1, 0, 2).contiguous()
    topk_idx_i32 = topk_idx.to(torch.int32).contiguous()   # wrapper requires int32
    topk_len_bq = topk_len.to(torch.int32).contiguous()    # (B, q_len)

    for use_len in (True, False):
        try:
            out = DSA.sparse_indexer_score_recompute_wrapper(
                q_bf, k_bf, w_bf, topk_idx_i32,
                qhead_per_kv_head=H,                 # 64 q-heads / 1 shared kv-head
                topk_length=(topk_len_bq if use_len else None),
                topk_indices_global=False,           # per-batch-local ids
                stream=None,
            )
            predict = out["predict"].float()         # (B, q_len, topk)
        except Exception as e:
            print(f"[topk_length={use_len}] wrapper raised: {type(e).__name__}: {e}")
            continue

        # On valid slots the two should match; ref is exactly 0 on invalid slots.
        diff = (predict - ref).abs()
        max_diff = diff.max().item()
        # Where ref==0 (invalid), predict should also be ~0 if masking matches.
        inv_mass = predict[ref == 0].abs().max().item() if (ref == 0).any() else 0.0
        print(f"[topk_length={use_len}] max|predict-ref|={max_diff:.3e}  "
              f"max predict on invalid slots={inv_mass:.3e}")
        if max_diff < 2e-2 and inv_mass < 2e-2:
            print(f"[topk_length={use_len}] PASS")
            return 0
    print("No configuration matched the reference — inspect the printed diffs above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
