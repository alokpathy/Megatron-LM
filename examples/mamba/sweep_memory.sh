#!/bin/bash
#
# DSA memory-usage sweep: triton vs cudnn across parallelism configs.
#
# For every (backend x parallelism-config x seq-len) it launches
# train_hybrid_moe.sh for a few iterations, then classifies the run and records
# peak GPU memory (the "max reserved" figure) into a single CSV.
#
# QBLOCK FALLBACK: each combo first runs at the true code-default query-block size
# (512). If that run OOMs, it is retried once at qblock=256. If the 256 retry also
# OOMs / times out / fails, the combo is recorded as FAIL (no further fallback).
#
# Each run is classified as:
#   OK             -> finished >=1 iteration; max_reserved_MB / max_allocated_MB filled
#   OOM            -> log contains a CUDA out-of-memory / torch OOM error
#   KILLED         -> reached training but died before iter 1 produced a memory line
#                     (external SIGKILL/SIGINT, or timed out mid first-step compile)
#   TIMEOUT        -> exceeded PER_RUN_TIMEOUT wall-clock (treated as KILLED-by-us)
#   CONFIG_INVALID -> emulation dims not divisible (script refused to run)
#   LAUNCH_FAIL    -> torchrun/torch/module not available in this shell
#   SETUP_FAIL     -> died before reaching the training loop for some other reason
#   FAIL           -> qblock=256 fallback exhausted (default OOM'd, retry also failed)
#
# IMPORTANT: run this INSIDE the environment where `torchrun` exists (container /
# module load). A bare login shell has no torch and every run will be LAUNCH_FAIL.
#
# Usage:
#   bash examples/mamba/sweep_memory.sh
#   FORCE=1 bash examples/mamba/sweep_memory.sh          # re-run even if a log already exists
#   TRAIN_ITERS=8 PER_RUN_TIMEOUT=1800 bash examples/mamba/sweep_memory.sh
#
# Output:
#   experiments/memory_sweep.csv        one row per run
#   experiments/mem_logs/log_<name>.out full per-run log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_hybrid_moe.sh"

LOG_DIR="${REPO_ROOT}/experiments/mem_logs"
OUT="${REPO_ROOT}/experiments/memory_sweep.csv"

# ============================ SWEEP AXES (edit) ============================
# Backends to compare.
BACKENDS=(triton cudnn)

# Parallelism configs as "EMU_TP EMU_EP EMU_PP" triples. The global model dims in
# train_hybrid_moe.sh must be divisible by these or the run is CONFIG_INVALID:
#   TP divides 80 (heads/mamba), 10240 (ffn/moe-ffn), 12288 (shared-expert)  -> {1,2,4,5,8,10,16,...}
#   EP divides 512 (experts)                                                 -> {1,2,4,...,512}
#   PP divides 12/70/70 (attn/ssm/moe layers) -> only {1,2} (gcd(12,70)=2)
# Lowering TP is the main memory-INCREASING lever (bigger per-GPU tensors);
# raising EP/PP shrinks per-GPU memory. tp=4/ep128/pp2 already peaks ~232 GB.
CONFIGS=(
  "8 128 2"
  "8 128 1"
  "8 64  2"
  "4 128 2"
  "16 128 2"
  # --- push the limits: tp=2 (per-GPU tensors ~2x tp=4). EP ladder maps the
  #     OOM boundary (high EP fits more easily; low EP is the heaviest). ---
  "2 512 2"
  "2 256 2"
  "2 128 2"
)

# Sequence lengths to cross with the above.
SEQ_LENS=(8192)

# DSA query-block fallback ladder. The first entry (512) is the true code default
# (min-memory kernel: _default_query_chunk_size = min(seqlen, 512)). On OOM the
# sweep steps to the next (smaller) entry; if the last entry still fails -> FAIL.
QBLOCK_LADDER=(512 256)

# Iterations per run (memory peaks by iter 2-3; >=4 is enough for a stable reading).
TRAIN_ITERS="${TRAIN_ITERS:-5}"

# Hard wall-clock cap per run (seconds). cuDNN's first-iteration compile is slow;
# give it room. A run exceeding this is recorded as TIMEOUT.
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-1500}"

# Re-run a combo even if its log already exists (default: reuse & just re-parse).
FORCE="${FORCE:-0}"

# GPU free-memory threshold (MiB) to wait for between runs, so a straggler from
# the previous run can't make the next one falsely OOM.
GPU_FREE_WAIT_MB="${GPU_FREE_WAIT_MB:-8000}"
# ==========================================================================

mkdir -p "${LOG_DIR}"

# Fresh CSV header.
echo "backend,seq_len,emu_tp,emu_ep,emu_pp,qblock,train_iters,status,max_reserved_MB,max_allocated_MB,iters_reached,log" > "${OUT}"

# ---- wait for the GPUs to be (mostly) free before launching the next run ----
wait_for_free_gpus() {
    for _ in $(seq 1 60); do
        local maxused
        maxused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
        [ -z "${maxused}" ] && return 0            # no nvidia-smi -> don't block
        [ "${maxused}" -lt "${GPU_FREE_WAIT_MB}" ] && return 0
        echo "    ...waiting for GPUs to free (busiest=${maxused} MiB)"
        sleep 10
    done
    echo "    WARN: GPUs still busy after wait; launching anyway."
}

# ---- classify a finished run's log; echoes: status maxres maxalloc iters ----
parse_log() {
    local log="$1" rc="$2"
    local status="" mr="" ma="" iters="" lastmem
    iters=$(grep -oE 'iteration +[0-9]+/' "$log" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
    [ -z "${iters}" ] && iters=0

    if grep -qiE 'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory' "$log" 2>/dev/null; then
        status="OOM"
    elif grep -qE 'ERROR: cannot emulate' "$log" 2>/dev/null; then
        status="CONFIG_INVALID"
    elif grep -qiE 'torchrun: command not found|No module named torch|ModuleNotFoundError: No module named .torch' "$log" 2>/dev/null; then
        status="LAUNCH_FAIL"
    elif grep -q 'max reserved:' "$log" 2>/dev/null; then
        status="OK"
        # max reserved is monotonic -> the last Rank 0 line holds the peak.
        lastmem=$(grep -oE '\[Rank 0\][^[]*max reserved: [0-9.]+' "$log" | tail -1)
        mr=$(printf '%s' "$lastmem" | grep -oE 'max reserved: [0-9.]+' | grep -oE '[0-9.]+')
        ma=$(printf '%s' "$lastmem" | grep -oE 'max allocated: [0-9.]+' | grep -oE '[0-9.]+')
    elif [ "$rc" -eq 124 ]; then
        status="TIMEOUT"
    elif grep -q 'before the start of training step' "$log" 2>/dev/null; then
        status="KILLED"
    else
        status="SETUP_FAIL"
    fi
    echo "${status}|${mr}|${ma}|${iters}"
}

# ---- launch (or reuse) one run at a given qblock; echoes: status|mr|ma|iters|log
# Progress chatter goes to stderr so the captured stdout stays parseable.
attempt_run() {
    local backend="$1" seq="$2" tp="$3" ep="$4" pp="$5" qb="$6"
    local name log rc=0
    name="memsweep_${backend}_seq${seq}_tp${tp}_ep${ep}_pp${pp}_q${qb}"
    log="${LOG_DIR}/log_${name}.out"

    # Reuse an existing terminal-status log unless FORCE=1.
    if [ "${FORCE}" != "1" ] && [ -f "${log}" ] \
       && grep -qiE 'max reserved:|CUDA out of memory|OutOfMemoryError|ERROR: cannot emulate' "${log}" 2>/dev/null; then
        echo ">>> ${name}  (reusing existing log)" >&2
    else
        wait_for_free_gpus >&2
        echo ">>> ${name}  (launching, timeout ${PER_RUN_TIMEOUT}s)" >&2
        TRAIN_ITERS="${TRAIN_ITERS}" EMU_TP="${tp}" EMU_EP="${ep}" EMU_PP="${pp}" QBLOCK="${qb}" \
            timeout -k 30 "${PER_RUN_TIMEOUT}" \
            bash "${TRAIN_SCRIPT}" "${name}" 1 0 "${backend}" "${seq}" \
            &> "${log}"
        rc=$?
        # Reap any torchrun/worker stragglers so the next run starts clean.
        pkill -9 -f 'pretrain_mamba.py' 2>/dev/null
        sleep 5
    fi
    echo "$(parse_log "${log}" "${rc}")|${log}"
}

echo "=== DSA memory sweep ==="
echo "backends: ${BACKENDS[*]}"
echo "configs (tp ep pp): ${CONFIGS[*]}"
echo "seq_lens: ${SEQ_LENS[*]}   qblock ladder: ${QBLOCK_LADDER[*]}   train_iters: ${TRAIN_ITERS}"
echo "output: ${OUT}"
echo ""
printf "%-8s %-7s %-3s %-4s %-3s %-5s  %-14s %12s %13s\n" \
    backend seq tp ep pp qblk status max_reserved max_allocated

for backend in "${BACKENDS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    read -r tp ep pp <<< "${cfg}"
    for seq in "${SEQ_LENS[@]}"; do

        # qblock fallback ladder: start at the default (512); on OOM step down.
        status="" mr="" ma="" iters="" log="" qb=""
        nrungs=${#QBLOCK_LADDER[@]}
        for idx in "${!QBLOCK_LADDER[@]}"; do
            qb="${QBLOCK_LADDER[$idx]}"
            IFS='|' read -r status mr ma iters log \
                <<< "$(attempt_run "${backend}" "${seq}" "${tp}" "${ep}" "${pp}" "${qb}")"

            [ "${status}" = "OK" ] && break

            last=$(( idx == nrungs - 1 ))
            if [ "${status}" = "OOM" ] && [ "${last}" -eq 0 ]; then
                echo "    OOM at qblock=${qb}; retrying at qblock=${QBLOCK_LADDER[$((idx+1))]}"
                continue
            fi
            # Terminal: last rung's OOM/timeout/any-failure collapses to FAIL.
            [ "${last}" -eq 1 ] && status="FAIL"
            break
        done

        printf "%-8s %-7s %-3s %-4s %-3s %-5s  %-14s %12s %13s\n" \
            "${backend}" "${seq}" "${tp}" "${ep}" "${pp}" "${qb}" \
            "${status}" "${mr:-–}" "${ma:-–}"

        echo "${backend},${seq},${tp},${ep},${pp},${qb},${TRAIN_ITERS},${status},${mr},${ma},${iters},${log}" >> "${OUT}"
    done
  done
done

echo ""
echo "=== done. results: ${OUT} ==="
column -t -s, "${OUT}"
