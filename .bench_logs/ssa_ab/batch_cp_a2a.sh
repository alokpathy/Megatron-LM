#!/bin/bash
# CP=2 all-to-all: Triton min-memory vs CuTe, at 32K.
#
# Order matters. The split test runs first and alone: it is single-GPU, bit-exact, and pins the
# re-implemented autograd Function against the wheel's own fused call. If that fails, nothing
# downstream is worth reading, because every CP number would be measured against a path that
# already disagrees with the reference at CP=1.
set -u

MEGATRON_LM_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/atripathy/Megatron-LM
LOG_DIR="${MEGATRON_LM_DIR}/.bench_logs/ssa_ab"
WHEEL=/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/dingqingy/simplified-sparse-attention/0.1.0.dev14/simplified_sparse_attention-0.1.0.dev14-py3-none-any.whl

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/batch_cp_a2a_$(date +%H%M%S).driver.log") 2>&1

cd "${MEGATRON_LM_DIR}" || exit 1
echo "=== node $(hostname)  $(date -Is)"
echo "=== capability: $(python -c 'import torch;print(torch.cuda.get_device_capability())' 2>/dev/null | tail -1)"
echo "=== commit: $(git log --oneline -1)"

pip install --no-deps "${WHEEL}" 2>&1 | tail -1
python -c "import simplified_sparse_attention as s; print('ssa', s.__version__ if hasattr(s,'__version__') else 'ok')" || exit 1

if [ ! -r "${HOME}/.netrc" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "=== no wandb credentials: offline"
    export WANDB_MODE=offline WANDB_API_KEY=offline
fi

echo
echo "################ split path == fused path (single GPU, bit-exact)"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q --no-header -p no:cacheprovider \
    tests/unit_tests/transformer/experimental_attention_variant/test_ssa_cute_split.py 2>&1 | tail -25
SPLIT_RC=${PIPESTATUS[0]}
echo "=== split test rc=${SPLIT_RC}"
if [ "${SPLIT_RC}" != "0" ]; then
    echo "=== split test failed; CP numbers would be measured against a broken reference. Stopping."
    exit 1
fi

echo
echo "################ CP=2 a2a, 32K, both arms, both orders"
for ORDER in "min-memory-triton ssa-cute" "ssa-cute min-memory-triton"; do
    echo "=== order='${ORDER}' start $(date -Is)"
    NGPU=4 CP=2 CP_COMM_TYPE=a2a \
        bash "${LOG_DIR}/run_profile.sh" 32768 "${ORDER}" 16
    echo "=== order='${ORDER}' done rc=$? $(date -Is)"
done

echo "=== all done $(date -Is)"
