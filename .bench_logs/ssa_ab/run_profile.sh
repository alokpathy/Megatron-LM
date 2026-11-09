#!/bin/bash
# Both kernel arms, profiled, inside one allocation.
#
#   bash run_profile.sh <seq_len> <arm_order> [iters]
#     arm_order: "min-memory-triton ssa-cute" or the reverse
#
# Whichever arm runs first pays allocation and page-cache warmup, which on this model has
# previously been larger than the effect under test. Run both orders and take the mean.
set -u

SEQ_LEN=${1:?usage: run_profile.sh <seq_len> <arm_order> [iters]}
ARMS=${2:?usage: run_profile.sh <seq_len> <arm_order> [iters]}
ITERS=${3:-16}

MEGATRON_LM_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/atripathy/Megatron-LM
LOG_DIR="${MEGATRON_LM_DIR}/.bench_logs/ssa_ab"
ORDER_TAG=$(echo "${ARMS}" | tr ' ' '-' | cut -c1-24)
LOG="${LOG_DIR}/profile_${SEQ_LEN}_${ORDER_TAG}.log"
mkdir -p "${LOG_DIR}"

cd "${MEGATRON_LM_DIR}" || exit 1

# wandb's console capture tees stdout through a socket. When the launching srun step is torn
# down that socket breaks, every later print raises BrokenPipeError inside a logging handler,
# and the run goes dark while still training. Off means stdout goes straight to the file.
export WANDB_CONSOLE=off

# GPUS and VISIBLE default to the two devices without leaked contexts; override for a clean node.
NGPU=${NGPU:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,3}

PROFILE=1 \
PROFILE_ARMS="${ARMS}" \
NO_SRUN=1 \
GPUS_PER_NODE="${NGPU}" \
LAUNCHER="python -m torch.distributed.run --nproc-per-node=${NGPU} --master-port=$((29000 + RANDOM % 900))" \
MEGATRON_LM_DIR="${MEGATRON_LM_DIR}" \
DIST_TIMEOUT_MIN=120 \
NSYS=0 \
SEQ_LEN="${SEQ_LEN}" CP=1 TP=2 ITERS="${ITERS}" GLOBAL_BATCH_SIZE=4 \
RUN_TAG="_prof${SEQ_LEN}_${ORDER_TAG}" \
WANDB_EXP_NAME="ssa_ab_${SEQ_LEN}_${ORDER_TAG}" \
    bash nt4_8b_ssa_ab.sh > "${LOG}" 2>&1

echo "=== exit=$? log=${LOG}"
grep -E '^ *iteration +[0-9]+/' "${LOG}" | tail -6
