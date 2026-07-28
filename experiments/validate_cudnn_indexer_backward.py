#!/usr/bin/env python
"""Standalone validation for cuDNN indexer_backward_wrapper (the DSA indexer WGRAD
core) vs the Megatron reference math. Touches NO training code.

Run inside the training container on a GPU node:
    python experiments/validate_cudnn_indexer_backward.py

The wrapper fuses the KL-gradient step with the score-backward, taking the teacher
(attn_score) and student (index_score) scores and returning activation gradients
d_index_q, d_weights, d_index_k. It replaces triton_selected_index_scores_backward
plus the (p - teacher)*scale step; the transforms + linear wgrad stay in Megatron.

Reference (mirrors _selected_index_scores_backward_torch :989 with
grad_selected_scores = (softmax(index_score) - teacher) * grad_loss*loss_coeff/(B*S_q)):

Prints max|cuDNN - ref| for each of d_index_q, d_weights, d_index_k. Small = match.
"""
import sys
import torch

try:
    from cudnn import DSA
except Exception:
    from cudnn.deepseek_sparse_attention import DSA

NEG = -1e9  # finite stand-in for -inf masking so the kernel doesn't see NaNs


def build_case(B, q_len, S_k, H, D, topk, q_start, device, dtype, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    q_index = torch.randn(q_len, B, H, D, generator=g, device=device, dtype=dtype)
    full_k = torch.randn(S_k, B, D, generator=g, device=device, dtype=dtype)
    weights = torch.rand(q_len, B, H, generator=g, device=device, dtype=dtype) * 0.05
    topk_idx = torch.empty(B, q_len, topk, dtype=torch.long, device=device)
    for b in range(B):
        for q in range(q_len):
            qpos = q_start + q
            n_valid = min(topk, qpos + 1)
            valid = torch.randperm(qpos + 1, generator=g, device=device)[:n_valid]
            pad = torch.full((topk - n_valid,), min(qpos + 1, S_k - 1), device=device)
            topk_idx[b, q] = torch.cat([valid, pad]).sort().values
    # random teacher distribution over VALID top-k slots (0 on invalid)
    qpos = torch.arange(q_start, q_start + q_len, device=device).view(1, q_len, 1)
    invalid = topk_idx > qpos                                   # (B, q_len, topk)
    t = torch.rand(B, q_len, topk, generator=g, device=device, dtype=torch.float32)
    t = t.masked_fill(invalid, 0.0)
    teacher = t / t.sum(dim=-1, keepdim=True).clamp_min(1e-9)   # (B, q_len, topk)
    return q_index, full_k, weights, topk_idx, teacher, invalid


def gather_selected_k(full_k, topk_idx):
    S_k, B, D = full_k.shape
    q_len, topk = topk_idx.shape[1], topk_idx.shape[2]
    k_bhd = full_k.permute(1, 0, 2)                             # (B, S_k, D)
    idx = topk_idx.long().clamp(0, S_k - 1)
    return torch.gather(
        k_bhd.unsqueeze(1).expand(B, q_len, S_k, D), 2,
        idx.unsqueeze(-1).expand(B, q_len, topk, D),
    )                                                          # (B, q_len, topk, D)


def reference(q_index, full_k, weights, topk_idx, teacher, invalid,
              loss_coeff, grad_loss, q_start):
    B, q_len, topk = topk_idx.shape
    sel_k = gather_selected_k(full_k, topk_idx).float()        # (B, q_len, topk, D)
    q = q_index.permute(1, 0, 2, 3).float()                    # (B, q_len, H, D)
    w = weights.permute(1, 0, 2).float()                       # (B, q_len, H)

    dot = torch.einsum("bqhd,bqkd->bqhk", q, sel_k)            # (B, q_len, H, topk)
    relu_dot = torch.relu(dot)
    index_score = (relu_dot * w.unsqueeze(-1)).sum(dim=2)      # (B, q_len, topk) student logits
    index_score_masked = index_score.masked_fill(invalid, NEG)
    p = torch.softmax(index_score_masked, dim=-1)             # (B, q_len, topk)

    scale = grad_loss * loss_coeff / (B * q_len)
    grad_ss = (p - teacher) * scale
    grad_ss = grad_ss.masked_fill(invalid, 0.0)               # (B, q_len, topk)

    relu_mask = (dot > 0).float()
    grad_weights = (grad_ss.unsqueeze(2) * relu_dot).sum(dim=-1)          # (B, q_len, H)
    grad_dot = grad_ss.unsqueeze(2) * w.unsqueeze(-1) * relu_mask         # (B, q_len, H, topk)
    grad_q_index = torch.einsum("bqhk,bqkd->bqhd", grad_dot, sel_k)       # (B, q_len, H, D)
    grad_selected_k = torch.einsum("bqhk,bqhd->bqkd", grad_dot, q)        # (B, q_len, topk, D)

    # scatter selected-k grad back to full (B, S_k, D)
    S_k = full_k.shape[0]
    grad_k_full = torch.zeros(B, S_k, full_k.shape[-1], device=q.device, dtype=torch.float32)
    for b in range(B):
        grad_k_full[b].index_add_(
            0, topk_idx[b].reshape(-1).long(), grad_selected_k[b].reshape(-1, full_k.shape[-1])
        )
    return {
        "index_score_masked": index_score_masked,   # raw student logits (masked)
        "p": p,                                       # student distribution (softmax'd)
        "d_index_q": grad_q_index,                   # (B, q_len, H, D)
        "d_weights": grad_weights,                   # (B, q_len, H)
        "d_index_k_full": grad_k_full,               # (B, S_k, D)
        "d_index_k_sel": grad_selected_k,            # (B, q_len, topk, D)
    }


def report(name, a, b):
    if a is None:
        print(f"  {name}: cuDNN output missing"); return
    if a.shape != b.shape:
        print(f"  {name}: SHAPE cuDNN {tuple(a.shape)} vs ref {tuple(b.shape)}"); return
    d = (a.float() - b.float()).abs().max().item()
    scale = b.float().abs().max().item() + 1e-9
    print(f"  {name}: max|diff|={d:.3e}  rel={d/scale:.3e}  {'PASS' if d/scale < 5e-2 else 'CHECK'}")


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA"); return 2
    dev = torch.device("cuda")
    B, q_len, S_k, H, D, topk, q_start = 2, 32, 256, 64, 128, 128, 0
    dtype = torch.bfloat16
    loss_coeff, grad_loss = 0.01, 1.0

    q_index, full_k, weights, topk_idx, teacher, invalid = build_case(
        B, q_len, S_k, H, D, topk, q_start, dev, dtype, seed=0)
    ref = reference(q_index, full_k, weights, topk_idx, teacher, invalid,
                    loss_coeff, grad_loss, q_start)

    # cuDNN layouts (bshd), fp32 scores; attn_score/index_score are consumed in-place -> clone.
    q_bf = q_index.permute(1, 0, 2, 3).contiguous()            # (B, q_len, H, D)
    k_bf = full_k.permute(1, 0, 2).contiguous()                # (B, S_k, D)
    w_bf = weights.permute(1, 0, 2).contiguous()               # (B, q_len, H)
    topk_i32 = topk_idx.to(torch.int32).contiguous()
    attn_score = teacher.clone().contiguous()                  # teacher distribution t
    index_score = ref["p"].clone().contiguous()                # student distribution p (softmax'd)

    try:
        out = DSA.indexer_backward_wrapper(
            q_bf, w_bf, k_bf,
            attn_score, index_score, topk_i32,
            sm_scale=1.0,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            topk_indices_global=False,
        )
    except Exception as e:
        print(f"wrapper raised: {type(e).__name__}: {e}")
        return 1

    d_q = out.get("d_index_q")
    d_w = out.get("d_weights")
    d_k = out.get("d_index_k")
    print("comparing cuDNN indexer_backward vs reference:")
    # d_index_q ref is (B, q_len, H, D); cuDNN likely same bshd
    report("d_index_q", d_q, ref["d_index_q"])
    report("d_weights", d_w, ref["d_weights"])
    # d_index_k: try full (B,S_k,D) first, then selected
    if d_k is not None and d_k.shape == ref["d_index_k_full"].shape:
        report("d_index_k[full]", d_k, ref["d_index_k_full"])
    else:
        report("d_index_k[sel]", d_k, ref["d_index_k_sel"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
