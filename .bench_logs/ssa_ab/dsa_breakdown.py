#!/usr/bin/env python3
"""Per-DSA-layer GPU time from a PyTorch-profiler trace, split indexer / attention / misc.

Usage:
    python3 -I dsa_breakdown.py <trace.json.gz> [--layers N] [--label NAME]
    python3 -I dsa_breakdown.py --compare <triton_trace> <cute_trace> [--layers N]

Reports the median over profiled steps, because a single step mixes autotune and warmup into
the number -- an earlier comparison on this model reversed sign depending on which step was
read.

Both arms run the same model, so "per DSA layer invocation" is total DSA GPU time divided by
(dsa_layers * microbatches). Pass --layers if the pattern is not the 4-attention-layer default.

Kernel names are matched case-insensitively against substring rules. Anything a rule does not
claim is listed under UNCLAIMED rather than silently folded into misc: a kernel this script has
never seen is exactly the thing that would make a comparison wrong without looking wrong.
"""

import argparse
import collections
import gzip
import json
import statistics
import sys

# (bucket, label, substrings). First match wins, so order matters.
RULES = [
    # --- Triton min-memory backend -------------------------------------------------
    ("INDEXER", "score (routing)", ["_dsa_simplified_score_block_kernel"]),
    ("INDEXER", "score (teacher)", ["_dsa_teacher_scores_kernel"]),
    ("INDEXER", "score bwd (qk)", ["_dsa_simplified_selected_scores_backward_qk"]),
    ("INDEXER", "proj wgrad", ["_dsa_linear_wgrad_kernel",
                               "_dsa_simplified_gathered_linear_wgrad"]),
    ("INDEXER", "norm stats", ["_dsa_simplified_input_norm_stats"]),
    ("INDEXER", "topk / select", ["mbtopk", "radix_select", "_dsa_topk", "select_kernel"]),
    ("INDEXER", "scatter grad", ["scatter_selected_grad"]),
    ("ATTENTION", "attn fwd", ["_dsa_sparse_attention_forward_kernel"]),
    ("ATTENTION", "attn bwd", ["_dsa_sparse_attention_backward_pair_kernel"]),
    # --- simplified_sparse_attention CuTe backend ----------------------------------
    # Named from the wheel's kernel modules; see simplified_sparse_attention/kernels/.
    ("INDEXER", "cute indexer fwd+topk", ["indexer_forward_topk"]),
    ("INDEXER", "cute indexer aux grad", ["indexer_aux_grad_precompute"]),
    ("INDEXER", "cute topk select", ["gvr_topk", "filtered_top_k", "block_scan"]),
    ("INDEXER", "cute dense indexer kl", ["phase1_dense_indexer_kl"]),
    ("ATTENTION", "cute attn fwd", ["sparse_attention_h96", "sparse_attention_large",
                                    "attention_teacher_forward"]),
    ("ATTENTION", "cute attn bwd", ["sparse_attention_backward"]),
    ("ATTENTION", "cute dense gemm", ["dense_gemm_sm100"]),
]

# Kernels that are DSA-adjacent but neither indexer nor attention: the projections feeding
# them, layout moves, and the loss bookkeeping.
MISC_RULES = [
    ("misc", ["dsa_", "simplified_sparse", "cute_dsl", "ssa_"]),
]


def classify(name):
    low = name.lower()
    for bucket, label, needles in RULES:
        if any(n.lower() in low for n in needles):
            return bucket, label
    for label, needles in MISC_RULES:
        if any(n.lower() in low for n in needles):
            return "MISC", label
    return None, None


def load_events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    return trace.get("traceEvents", trace if isinstance(trace, list) else [])


def step_windows(events):
    """[(start_us, end_us)] for each ProfilerStep range."""
    windows = []
    for event in events:
        name = event.get("name", "")
        if name.startswith("ProfilerStep") and event.get("ph") == "X":
            start = event.get("ts", 0)
            windows.append((start, start + event.get("dur", 0)))
    return sorted(windows)


def gpu_kernels(events):
    """Device-side kernel events: ph=X on a kernel category."""
    out = []
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = (event.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        out.append((event.get("ts", 0), event.get("dur", 0) or 0, event.get("name", "")))
    return out


def analyse(path, layers, microbatches):
    events = load_events(path)
    windows = step_windows(events)
    kernels = gpu_kernels(events)
    if not windows:
        # No ProfilerStep markers: treat the whole trace as one window.
        windows = [(min((k[0] for k in kernels), default=0),
                    max((k[0] + k[1] for k in kernels), default=0))]

    per_step = []
    unclaimed = collections.Counter()
    for start, end in windows:
        totals = collections.Counter()
        for ts, dur, name in kernels:
            if ts < start or ts >= end:
                continue
            bucket, label = classify(name)
            if bucket is None:
                if "dsa" in name.lower() or "cute" in name.lower():
                    unclaimed[name] += dur
                continue
            totals[(bucket, label)] += dur
        if totals:
            per_step.append(totals)

    if not per_step:
        return None, unclaimed, len(windows)

    labels = sorted({key for step in per_step for key in step})
    median = {key: statistics.median([step.get(key, 0) for step in per_step]) for key in labels}
    return median, unclaimed, len(windows)


def report(label, median, unclaimed, n_windows, layers, microbatches):
    invocations = layers * microbatches
    print(f"\n=== {label}  ({n_windows} profiled step(s), "
          f"{layers} DSA layers x {microbatches} microbatches = {invocations} invocations)")
    if median is None:
        print("  no DSA kernels matched in any step")
        return {}
    by_bucket = collections.Counter()
    for (bucket, _), value in median.items():
        by_bucket[bucket] += value
    grand = sum(by_bucket.values())
    print(f"  {'bucket / kernel':<34}{'ms/step':>10}{'ms/layer':>11}{'% of DSA':>10}")
    print("  " + "-" * 63)
    for bucket in ("INDEXER", "ATTENTION", "MISC"):
        if not by_bucket.get(bucket):
            continue
        total = by_bucket[bucket]
        print(f"  {bucket:<34}{total/1000:>10.2f}{total/1000/invocations:>11.3f}"
              f"{100*total/grand:>9.1f}%")
        for (b, lab), value in sorted(median.items(), key=lambda kv: -kv[1]):
            if b != bucket:
                continue
            print(f"    {lab:<32}{value/1000:>10.2f}{value/1000/invocations:>11.3f}"
                  f"{100*value/grand:>9.1f}%")
    print("  " + "-" * 63)
    print(f"  {'TOTAL DSA':<34}{grand/1000:>10.2f}{grand/1000/invocations:>11.3f}{100.0:>9.1f}%")
    if unclaimed:
        print("  UNCLAIMED (DSA-looking kernels no rule matched):")
        for name, value in unclaimed.most_common(12):
            print(f"    {name[:60]:<60}{value/1000:>9.2f} ms")
    return by_bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--layers", type=int, default=4, help="DSA attention layers per model")
    parser.add_argument("--microbatches", type=int, default=2, help="microbatches per step")
    parser.add_argument("--labels", default=None, help="comma-separated names for the traces")
    args = parser.parse_args()

    labels = args.labels.split(",") if args.labels else args.traces
    results = {}
    for path, label in zip(args.traces, labels):
        median, unclaimed, n = analyse(path, args.layers, args.microbatches)
        results[label] = report(label, median, unclaimed, n, args.layers, args.microbatches)

    if len(results) == 2:
        (name_a, a), (name_b, b) = results.items()
        print(f"\n=== {name_a} vs {name_b} (per step, ms)")
        print(f"  {'bucket':<14}{name_a[:16]:>17}{name_b[:16]:>17}{'delta':>12}{'change':>10}")
        for bucket in ("INDEXER", "ATTENTION", "MISC"):
            x, y = a.get(bucket, 0) / 1000, b.get(bucket, 0) / 1000
            if not x and not y:
                continue
            pct = f"{100*(y-x)/x:+.1f}%" if x else "n/a"
            print(f"  {bucket:<14}{x:>17.2f}{y:>17.2f}{y-x:>12.2f}{pct:>10}")
        x, y = sum(a.values()) / 1000, sum(b.values()) / 1000
        pct = f"{100*(y-x)/x:+.1f}%" if x else "n/a"
        print(f"  {'TOTAL':<14}{x:>17.2f}{y:>17.2f}{y-x:>12.2f}{pct:>10}")


if __name__ == "__main__":
    sys.exit(main())
