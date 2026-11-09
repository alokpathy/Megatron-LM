#!/bin/bash
# Batch entry point for the Triton-vs-CuTe DSA kernel A/B.
#
# Detached (sbatch) rather than driven interactively: every interactive attempt died when the
# launching step was torn down before python had been exec'd. A batch job owns its own
# allocation, so nothing external can abort it mid-launch.
#
#   bash batch_entry.sh [seq_lens...]      default: 8192 32768
#
# Runs both arms, both orders, for each sequence length. Whichever arm runs first pays
# allocation and page-cache warmup, which on this model has previously exceeded the effect
# under test and reversed its sign, so a single order is not a result.
set -u

MEGATRON_LM_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/atripathy/Megatron-LM
LOG_DIR="${MEGATRON_LM_DIR}/.bench_logs/ssa_ab"
WHEEL=/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/dingqingy/simplified-sparse-attention/0.1.0.dev14/simplified_sparse_attention-0.1.0.dev14-py3-none-any.whl
STAMP="${LOG_DIR}/batch_$(date +%H%M%S)"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${STAMP}.driver.log") 2>&1

echo "=== node $(hostname)  $(date -Is)"
echo "=== capability: $(python -c 'import torch;print(torch.cuda.get_device_capability())' 2>/dev/null | tail -1)"

# The container is fresh, so the wheel from any earlier session is gone.
pip install --no-deps "${WHEEL}" 2>&1 | tail -2
python -c "import simplified_sparse_attention as s; print('ssa', s.__file__)" || exit 1

# The launcher exits if it can find no wandb key. Fall back to offline rather than lose the
# timings; an offline run can be `wandb sync`ed afterwards.
if [ ! -r "${HOME}/.netrc" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "=== no ~/.netrc and no WANDB_API_KEY: running wandb offline"
    export WANDB_MODE=offline WANDB_API_KEY=offline
fi

SEQ_LENS=("$@")
[ ${#SEQ_LENS[@]} -eq 0 ] && SEQ_LENS=(8192 32768)

for SEQ in "${SEQ_LENS[@]}"; do
    for ORDER in "min-memory-triton ssa-cute" "ssa-cute min-memory-triton"; do
        echo "=== seq=${SEQ} order='${ORDER}' start $(date -Is)"
        NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
            bash "${LOG_DIR}/run_profile.sh" "${SEQ}" "${ORDER}" 16
        echo "=== seq=${SEQ} order='${ORDER}' done rc=$? $(date -Is)"
    done
done

echo "=== all done $(date -Is)"
