#!/bin/bash
# One arm of the Triton-vs-CuTe DSA kernel A/B, for the 4-GPU interactive node.
#
#   bash run_arm.sh <backend> <master_port>
#
# Writes the full training log beside this script and echoes the per-iteration loss lines so a
# blocking caller sees the result without reading the log.
set -u

BACKEND=${1:?usage: run_arm.sh <backend> <master_port>}
PORT=${2:?usage: run_arm.sh <backend> <master_port>}

MEGATRON_LM_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/atripathy/Megatron-LM
LOG_DIR="${MEGATRON_LM_DIR}/.bench_logs/ssa_ab"
LOG="${LOG_DIR}/${BACKEND}.log"
mkdir -p "${LOG_DIR}"

cd "${MEGATRON_LM_DIR}" || exit 1

DSA_BACKEND="${BACKEND}" \
NO_SRUN=1 \
LAUNCHER="python -m torch.distributed.run --nproc-per-node=4 --master-port=${PORT}" \
WANDB_MODE=offline WANDB_API_KEY=offline \
MEGATRON_LM_DIR="${MEGATRON_LM_DIR}" \
NSYS=0 PROFILE=0 SEQ_LEN=8192 CP=1 TP=2 ITERS=3 GLOBAL_BATCH_SIZE=4 \
RUN_TAG="_${BACKEND}_smoke" \
    bash nt4_8b_ssa_ab.sh > "${LOG}" 2>&1
STATUS=$?

echo "=== ${BACKEND} exit=${STATUS} log=${LOG}"
echo "=== topology"
grep -m1 '^world=' "${LOG}"
echo "=== iteration lines"
grep ' iteration ' "${LOG}" | tail -5
echo "=== last lines"
tail -25 "${LOG}"
exit ${STATUS}
