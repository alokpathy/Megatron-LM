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
CP=${CP:-1}
CP_COMM_TYPE=${CP_COMM_TYPE:-all_gather}
ORDER_TAG=$(echo "${ARMS}" | tr ' ' '-' | cut -c1-24)
# CP and the comm type are part of the identity of a run, not just its shape: two arms that
# share a run directory have the second resume the first's checkpoint and train nothing.
TAG="${SEQ_LEN}_cp${CP}${CP_COMM_TYPE:+_${CP_COMM_TYPE}}_${ORDER_TAG}"
LOG="${LOG_DIR}/profile_${TAG}.log"
mkdir -p "${LOG_DIR}"

cd "${MEGATRON_LM_DIR}" || exit 1

# wandb's console capture tees stdout through a socket. When the launching srun step is torn
# down that socket breaks, every later print raises BrokenPipeError inside a logging handler,
# and the run goes dark while still training. Off means stdout goes straight to the file.
export WANDB_CONSOLE=off

NGPU=${NGPU:-4}

# A rerun has to start from nothing. Each arm saves at exit, and on the next run --load finds
# that checkpoint, resumes at the final iteration and exits with nothing to train: no error, no
# trace, and an arm silently missing from the comparison. This bit once already, when a rerun
# after a crash left the arm that had succeeded with a checkpoint and the arm that had failed
# without one. KEEP_RUN_DIR=1 to opt out.
RUN_ROOT="/lustre/fsw/portfolios/nemotron/users/atripathy/runs/dsa_gqa_ssa_ab"
if [ "${KEEP_RUN_DIR:-0}" != "1" ]; then
    for stale in "${RUN_ROOT}"/*"_prof${TAG}"; do
        [ -d "${stale}" ] && echo "removing stale run dir: ${stale}" && rm -rf "${stale}"
    done
fi

PROFILE=1 \
PROFILE_ARMS="${ARMS}" \
NO_SRUN=1 \
GPUS_PER_NODE="${NGPU}" \
LAUNCHER="python -m torch.distributed.run --nproc-per-node=${NGPU} --master-port=$((29000 + RANDOM % 900))" \
MEGATRON_LM_DIR="${MEGATRON_LM_DIR}" \
DIST_TIMEOUT_MIN=120 \
NSYS=0 \
SEQ_LEN="${SEQ_LEN}" CP="${CP}" CP_COMM_TYPE="${CP_COMM_TYPE}" TP=2 \
ITERS="${ITERS}" GLOBAL_BATCH_SIZE=4 \
RUN_TAG="_prof${TAG}" \
WANDB_EXP_NAME="ssa_ab_${TAG}" \
    bash nt4_8b_ssa_ab.sh > "${LOG}" 2>&1

echo "=== exit=$? log=${LOG}"
grep -E '^ *iteration +[0-9]+/' "${LOG}" | tail -6
