#!/usr/bin/env python
"""Validate the full _cudnn_indexer_backward_wgrad helper (cuDNN indexer_backward
+ Q/weights/K tails) against the existing trusted _native_indexer_loss_wgrad_chunk.
Touches NO training code (imports the module functions, runs both paths, diffs the
5 accumulated weight gradients).

Run inside the training container on a GPU node:
    python experiments/validate_cudnn_indexer_backward_wgrad.py

RoPE/Hadamard are OFF here: this checks the PLUMBING (permutes, 4-D reshape,
weights scaling, wrapper I/O, no-scatter K tail). Position correctness (rope-on,
full-seq arange vs selected topk) is validated separately at fp64 in
experiments/validate_cudnn_ktail.py.
"""
import os
import sys
import torch

# Ensure the repo root (parent of experiments/) is importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

M = "megatron.core.transformer.experimental_attention_variant.dsa_min_memory"
mod = __import__(M, fromlist=["*"])
_project_q_index_tile = mod._project_q_index_tile
_project_k_index_block = mod._project_k_index_block
_gather_selected_indexer_k = mod._gather_selected_indexer_k
_index_scores_for_selected = mod._index_scores_for_selected
_selected_causal_invalid_mask = mod._selected_causal_invalid_mask
_native_indexer_loss_wgrad_chunk = mod._native_indexer_loss_wgrad_chunk
_cudnn_indexer_backward_wgrad = mod._cudnn_indexer_backward_wgrad


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA"); return 2
    dev = torch.device("cuda")
    torch.manual_seed(0)
    B, S, H, D, Hd, topk = 2, 128, 64, 128, 64, 128  # H>=64 required by the cuDNN kernel
    q_start, q_end = 0, S
    eps = 1e-5
    use_rope, use_had, rope_dim = False, False, 0
    rpe, rint = None, False
    loss_coeff, grad_loss = 0.01, 1.0
    dtype = torch.bfloat16

    hidden = torch.randn(S, B, Hd, device=dev, dtype=dtype)
    lqw = torch.randn(H * D, Hd, device=dev, dtype=dtype) * (Hd ** -0.5)
    lkw = torch.randn(D, Hd, device=dev, dtype=dtype) * (Hd ** -0.5)
    lww = torch.randn(H, Hd, device=dev, dtype=dtype) * (Hd ** -0.5)
    knw = torch.randn(D, device=dev, dtype=dtype)
    knb = torch.randn(D, device=dev, dtype=dtype)

    q_index, weights = _project_q_index_tile(
        hidden, q_start, q_end, lqw, lww, H, D, rope_dim, rpe, rint, use_rope, use_had, None, None)
    full_k = _project_k_index_block(
        hidden, 0, S, lkw, knw, knb, True, eps, D, rope_dim, rpe, rint, use_rope, use_had, None, None)

    # causal top-k indices (keys <= query pos), padded to qpos
    topk_idx = torch.empty(B, S, topk, dtype=torch.long, device=dev)
    for b in range(B):
        for q in range(S):
            n = min(topk, q + 1)
            valid = torch.randperm(q + 1, device=dev)[:n]
            pad = torch.full((topk - n,), min(q, S - 1), device=dev)
            topk_idx[b, q] = torch.cat([valid, pad]).sort().values

    # student p and teacher t (both distributions, 0 on invalid slots)
    sel_k = _gather_selected_indexer_k(full_k, topk_idx)
    logits = _index_scores_for_selected(q_index, weights, sel_k)          # (B, S, topk)
    invalid = _selected_causal_invalid_mask(topk_idx, q_start)
    p = torch.softmax(logits.float().masked_fill(invalid, float("-inf")), dim=-1)
    t = torch.rand(B, S, topk, device=dev, dtype=torch.float32).masked_fill(invalid, 0.0)
    t = t / t.sum(-1, keepdim=True).clamp_min(1e-9)

    scale = grad_loss * loss_coeff / (B * S)
    grad_ss = ((p - t) * scale).masked_fill(invalid, 0.0)

    def zeros():
        return [torch.zeros_like(x, dtype=torch.float32) for x in (lqw, lkw, knw, knb, lww)]

    # ----- reference: existing selected WGRAD -----
    r = zeros()
    ok = _native_indexer_loss_wgrad_chunk(
        hidden, q_start, q_end, topk_idx, q_index, weights, grad_ss,
        lqw, lkw, knw, knb, True, lww, eps, D, rope_dim, rpe, rint, use_rope, use_had,
        r[0], r[1], r[2], r[3], r[4], None)
    if not ok:
        print("reference _native_indexer_loss_wgrad_chunk returned False (triton unavailable?)"); return 1

    # ----- candidate: cuDNN fused WGRAD -----
    c = zeros()
    ok2 = _cudnn_indexer_backward_wgrad(
        hidden, q_start, q_end, topk_idx, q_index, weights, full_k, t, p,
        lkw, knw, knb, True, eps, H, D, rope_dim, rpe, rint, use_rope, use_had,
        c[0], c[1], c[2], c[3], c[4], loss_coeff, grad_loss)
    if not ok2:
        print("_cudnn_indexer_backward_wgrad returned False (cuDNN kernel unavailable?)"); return 1

    names = ["grad_linear_q_weight", "grad_linear_k_weight",
             "grad_k_norm_weight", "grad_k_norm_bias", "grad_linear_weights_weight"]
    print("cuDNN fused WGRAD vs selected reference:")
    allpass = True
    for n, a, b in zip(names, c, r):
        d = (a - b).abs().max().item()
        s = b.abs().max().item() + 1e-30
        ok = d / s < 3e-2
        allpass = allpass and ok
        print(f"  {n}: max|diff|={d:.3e}  rel={d/s:.3e}  {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if allpass else "SOME FAILED")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
