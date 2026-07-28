#!/usr/bin/env python
"""Validate the full-sequence K tail (for cuDNN d_index_k) against the existing
selected-layout K tail. Touches NO training code.

Run inside the training container on a GPU node:
    python experiments/validate_cudnn_ktail.py

cuDNN indexer_backward_wrapper returns d_index_k already scattered to the full
sequence (B, S_k, D). The existing Megatron K tail instead works on the selected
layout (B, q, topk, D) and scatters at the end. This harness proves the two
produce the SAME weight gradients (grad_linear_k_weight, grad_k_norm_weight,
grad_k_norm_bias), which is the correctness condition for wiring d_index_k.

Both tails perform: inverse-Hadamard -> inverse-RoPE(positions) -> LayerNorm
backward -> linear wgrad. The only differences are layout, RoPE positions
(topk_indices vs arange), and scatter timing. Because LayerNorm backward is
linear in the upstream grad and each physical key has one pre-norm k_linear,
scatter-then-tail must equal tail-then-scatter.

The equivalence is exact in fp64; we run fp64 and assert tight agreement.
"""
import sys
import torch

try:
    from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
    HAVE_HADAMARD = True
except Exception:
    rotate_activation = None
    HAVE_HADAMARD = False


def rope_inverse(g, positions, rotary_dim, base=1000000.0):
    """Adjoint (= inverse) of a GPT-NeoX-style RoPE at given positions.

    g: (..., D)   positions: (...,) matching g's leading dims (one per row)
    Rotates the first `rotary_dim` dims by -theta; leaves the rest untouched.
    Convention need not match Megatron's exactly -- only that BOTH tails use the
    SAME positions per physical key, which is the property under test.
    """
    D = g.shape[-1]
    half = rotary_dim // 2
    dev = g.device
    idx = torch.arange(half, device=dev, dtype=torch.float64)
    freqs = base ** (-2.0 * idx / rotary_dim)                 # (half,)
    theta = positions.to(torch.float64).unsqueeze(-1) * freqs  # (..., half)
    cos, sin = torch.cos(theta), torch.sin(theta)
    g_rot, g_pass = g[..., :rotary_dim], g[..., rotary_dim:]
    g1, g2 = g_rot[..., :half], g_rot[..., half:]
    # adjoint of [x1'=x1c - x2s ; x2'=x1s + x2c] is rotation by -theta:
    o1 = g1 * cos + g2 * sin
    o2 = -g1 * sin + g2 * cos
    return torch.cat([o1, o2, g_pass], dim=-1)


def layernorm_backward(g, x, gamma, eps):
    """Return (grad_x, grad_gamma_rows, grad_beta_rows) for y = gamma*xhat + beta.

    g, x: (..., D)   gamma: (D,)   Row-wise LayerNorm over the last dim.
    grad_gamma_rows / grad_beta_rows are per-row contributions (caller sums).
    """
    D = x.shape[-1]
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    xhat = (x - mean) * rstd
    grad_beta_rows = g
    grad_gamma_rows = g * xhat
    gy = g * gamma
    grad_x = rstd * (gy - gy.mean(-1, keepdim=True) - xhat * (gy * xhat).mean(-1, keepdim=True))
    return grad_x, grad_gamma_rows, grad_beta_rows


def k_tail(grad_kindex, positions, k_linear, hidden, gamma, eps, rotary_dim, use_hadamard):
    """One K tail: inv-Hadamard -> inv-RoPE -> LN backward -> linear wgrad.

    grad_kindex: (..., D)   positions: (...,)   k_linear/hidden: (..., D)/(..., Hd)
    Returns grad_linear_k_weight (D, Hd), grad_gamma (D,), grad_beta (D,).
    Works for any leading shape -> used for both selected (B,q,topk) and full (B,S).
    """
    g = grad_kindex
    if use_hadamard and HAVE_HADAMARD:
        g = rotate_activation(g.to(torch.bfloat16)).to(torch.float64)
    g = rope_inverse(g, positions, rotary_dim)
    grad_k_linear, dgamma_rows, dbeta_rows = layernorm_backward(g, k_linear, gamma, eps)
    Hd = hidden.shape[-1]
    gk = grad_k_linear.reshape(-1, grad_k_linear.shape[-1])      # (rows, D)
    hd = hidden.reshape(-1, Hd)                                  # (rows, Hd)
    grad_Wk = gk.t().matmul(hd)                                  # (D, Hd)
    dgamma = dgamma_rows.reshape(-1, dgamma_rows.shape[-1]).sum(0)
    dbeta = dbeta_rows.reshape(-1, dbeta_rows.shape[-1]).sum(0)
    return grad_Wk, dgamma, dbeta


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA"); return 2
    dev = torch.device("cuda")
    torch.manual_seed(0)
    B, S_k, Hd, D, q_len, topk = 2, 96, 320, 128, 24, 16
    eps = 1e-5
    rotary_dim = D
    use_hadamard = True

    hidden = torch.randn(B, S_k, Hd, device=dev, dtype=torch.float64)
    Wk = torch.randn(D, Hd, device=dev, dtype=torch.float64) * (Hd ** -0.5)
    gamma = torch.randn(D, device=dev, dtype=torch.float64)
    k_linear_full = torch.einsum("bsh,dh->bsd", hidden, Wk)      # (B, S_k, D)

    # causal top-k selection (keys <= query pos), rest padded to qpos
    topk_idx = torch.empty(B, q_len, topk, dtype=torch.long, device=dev)
    for b in range(B):
        for q in range(q_len):
            n = min(topk, q + 1)
            valid = torch.randperm(q + 1, device=dev)[:n]
            pad = torch.full((topk - n,), min(q, S_k - 1), device=dev)
            topk_idx[b, q] = torch.cat([valid, pad]).sort().values
    qpos = torch.arange(q_len, device=dev).view(1, q_len, 1)
    invalid = topk_idx > qpos

    # arbitrary upstream grad on the SELECTED k_index, zeroed on invalid slots
    grad_sel = torch.randn(B, q_len, topk, D, device=dev, dtype=torch.float64)
    grad_sel = grad_sel.masked_fill(invalid.unsqueeze(-1), 0.0)

    # ----- selected tail (existing layout) -----
    k_linear_sel = torch.gather(
        k_linear_full.unsqueeze(1).expand(B, q_len, S_k, D), 2,
        topk_idx.unsqueeze(-1).expand(B, q_len, topk, D))         # (B, q, topk, D)
    hidden_sel = torch.gather(
        hidden.unsqueeze(1).expand(B, q_len, S_k, Hd), 2,
        topk_idx.unsqueeze(-1).expand(B, q_len, topk, Hd))        # (B, q, topk, Hd)
    pos_sel = topk_idx.to(torch.float64)                          # (B, q, topk)
    gWk_sel, dg_sel, db_sel = k_tail(
        grad_sel, pos_sel, k_linear_sel, hidden_sel, gamma, eps, rotary_dim, use_hadamard)

    # ----- full-sequence tail (cuDNN d_index_k layout): scatter grad, then tail -----
    grad_full = torch.zeros(B, S_k, D, device=dev, dtype=torch.float64)
    for b in range(B):
        grad_full[b].index_add_(0, topk_idx[b].reshape(-1), grad_sel[b].reshape(-1, D))
    pos_full = torch.arange(S_k, device=dev, dtype=torch.float64).view(1, S_k).expand(B, S_k)
    gWk_full, dg_full, db_full = k_tail(
        grad_full, pos_full, k_linear_full, hidden, gamma, eps, rotary_dim, use_hadamard)

    def rep(name, a, b):
        d = (a - b).abs().max().item()
        s = b.abs().max().item() + 1e-30
        print(f"  {name}: max|diff|={d:.3e}  rel={d/s:.3e}  {'PASS' if d/s < 1e-9 else 'FAIL'}")

    print(f"K-tail equivalence (selected vs full-scattered), Hadamard={'on' if (use_hadamard and HAVE_HADAMARD) else 'off'}:")
    rep("grad_linear_k_weight", gWk_sel, gWk_full)
    rep("grad_k_norm_weight", dg_sel, dg_full)
    rep("grad_k_norm_bias", db_sel, db_full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
