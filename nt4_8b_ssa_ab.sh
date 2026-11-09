#!/bin/bash
# Triton vs CuTe kernel A/B for DSA over GQA, derived from nt4_8b_cp_ab.sh.
#
# This is a *kernel* benchmark, not a convergence run. The model is reshaped to the CuTe
# kernels' fixed envelope, so it is not the production 8B and its checkpoints do not load:
#
#   kv-channels           192 -> 256   (attention head dim the CuTe kernels require)
#   dsa-indexer-head-dim  192 -> 128   (indexer dim the CuTe kernels require)
#
# Everything else matches nt4_8b_cp_ab.sh. Because both dims changed, step times here are not
# comparable with any measurement from that script -- only the two arms below are comparable
# with each other.
#
# Submit one arm:
#   export WANDB_API_KEY=...
#   DSA_BACKEND=min-memory-triton sbatch --job-name=ssa_triton nt4_8b_ssa_ab.sh
#   DSA_BACKEND=ssa-cute          sbatch --job-name=ssa_cute   nt4_8b_ssa_ab.sh
#
# Or interleave both inside one allocation, which is what the timings should come from --
# at this scale the difference between two allocations is larger than the difference between
# the arms:
#   PROFILE=1 PROFILE_ARMS="min-memory-triton ssa-cute" sbatch nt4_8b_ssa_ab.sh
#   PROFILE=1 PROFILE_ARMS="ssa-cute min-memory-triton" sbatch nt4_8b_ssa_ab.sh
#
# Run both orderings and take the mean. Whichever arm runs first pays an allocation-warmup
# cost that has previously been larger than the effect under test and flipped its sign.
#
# Requirements for the ssa-cute arm, all enforced in transformer_config.py:
#   - compute capability (10, 3); SM100 (GB200) is rejected inside the wheel
#   - simplified_sparse_attention installed: pip install --no-deps <wheel>
#   - topk in {512, 1024, 2048}, heads in {16, 32, 96}, one KV group, BF16
#   - no CUDA graphs, and no --dsa-fwd-use-dense-attn (there is no dense warmup path)

#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --ntasks-per-node=4
#SBATCH --segment=4
#SBATCH --nodes=12
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0

set -euo pipefail

export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork

export QUANTIZATION_TYPE_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SHM_DISABLE=1

# GB300 is Blackwell: CUDA_DEVICE_MAX_CONNECTIONS is not required and has no effect.

################################################################
### A/B knobs
################################################################
CP=${CP:-1}                       # context-parallel size
CP_LAYOUT=${CP_LAYOUT:-zigzag}    # zigzag (default) | contiguous
# all_gather moves every rank's K and V to every rank. a2a moves only the keys top-k
# selected, exchanged per query tile after routing has chosen them. Both are valid TE
# spellings, so the non-DSA attention layers accept either.
CP_COMM_TYPE=${CP_COMM_TYPE:-all_gather}
TP=${TP:-8}

# The dimension under test. min-memory-triton is the current production path; ssa-cute routes
# the same forward and backward into the simplified_sparse_attention wheel's CuTe kernels.
DSA_BACKEND=${DSA_BACKEND:-min-memory-triton}
case "${DSA_BACKEND}" in
    min-memory-triton|min-memory-torch|ssa-cute) ;;
    *)
        echo "ERROR: DSA_BACKEND='${DSA_BACKEND}' must be min-memory-triton, min-memory-torch or ssa-cute." >&2
        exit 1
        ;;
esac

# DRY_RUN validates the flag surface, not the topology. It runs one process, so validate_args
# sees world_size=1 and would reject the real TP/CP regardless; the divisibility of the real
# topology is checked arithmetically below instead. Force a 1x1 layout so parsing reaches the
# flags we actually want to prove still exist.
if [ "${DRY_RUN:-0}" = "1" ]; then
    TP=1
    CP=1
fi
ITERS=${ITERS:-200}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EXIT_MINS=${EXIT_MINS:-225}

WANDB_PROJECT=${WANDB_PROJECT:-dsa-gqa-cp}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-8b_dsa_gqa_cp${CP}_${CP_LAYOUT}}
# --wandb-entity is left unset so wandb uses the default entity for the API key.
# Set WANDB_ENTITY to override.

# The srun below uses --no-container-mount-home, so ~/.netrc is invisible inside the
# container. Read the key here on the batch node and pass it through as an env var.
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "${HOME}/.netrc" ]; then
    WANDB_API_KEY=$(awk '/machine[ \t]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' "${HOME}/.netrc")
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: no wandb credentials found." >&2
    echo "  Run 'wandb login' once (writes ~/.netrc), or export WANDB_API_KEY." >&2
    echo "  For an offline run instead: export WANDB_MODE=offline WANDB_API_KEY=offline" >&2
    exit 1
fi
export WANDB_API_KEY

################################################################
### Paths
################################################################
NAME="8B_ssa_kernel_ab_cp${CP}_${CP_LAYOUT}_${DSA_BACKEND}${RUN_TAG:-}"
BASE_DIR="/lustre/fsw/portfolios/nemotron/users/atripathy"
RUN_DIR="${BASE_DIR}/runs/dsa_gqa_ssa_ab/${NAME}"

# rwaleffe's image: the one known to run this exact DSA model config.
IMAGE="/lustre/fsw/portfolios/nemotron/users/rwaleffe/images/nvidia+pytorch+26.04-py3+dependencies+mamba+fast_hadamard+eval.sqsh"

# Must be the branch that HAS the CP-over-DSA-GQA work. rwaleffe's June-30 checkout
# does not (no context_parallel references in its experimental_attention_variant/).
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/atripathy/Megatron-LM}"

DATACACHE_DIR="${BASE_DIR}/runs/data-cache/"

TOKENIZER_MODEL_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/dnarayanan/nemotron4_prep/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"
BLEND_PATH="/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron6/blend_files/1T-phase1var-moresft.json"

# No phase toggle here. Both DSA checkpoints were trained at kv-channels 192 and indexer dim
# 192, so neither loads into this model -- every attention and indexer tensor has the wrong
# shape. Random init is also what a kernel A/B wants: both arms start from the same seed and
# the kernels see the same shapes either way.
#
# Phase 1 is unavailable for a second reason: --dsa-fwd-use-dense-attn has no ssa-cute path,
# so the arms would not be running the same thing.
PHASE_ARGS="--dsa-indexer-use-sparse-loss"
PHASE_LR="${LR:-1.28e-4}"

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-none}"

# Set PRETRAINED_CHECKPOINT=none to start from random weights. The checkpoint carries its own
# consumed-sample count, so a short schedule resumes past the end of the sampler and asserts
# with "no samples left to consume" before training starts. A profiling run does not care what
# the weights are -- the kernels and their shapes are the same either way -- and a short
# schedule is the whole point of one.
PRETRAINED_ARG="--pretrained-checkpoint ${PRETRAINED_CHECKPOINT}"
if [ "${PRETRAINED_CHECKPOINT}" = "none" ]; then
    PRETRAINED_ARG=""
fi

LOGS_DIR="${RUN_DIR}/logs/"
# Per-arm and initially empty, so --load falls through to --pretrained-checkpoint and
# both arms start from bit-identical weights.
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/"
WANDB_SAVE_DIR="${RUN_DIR}/wandb/"

mkdir -p ${LOGS_DIR} ${CHECKPOINT_DIR} ${DATACACHE_DIR} ${TENSORBOARD_DIR} ${WANDB_SAVE_DIR}

# Kernel caches. /home/tmp works under the srun below, which mounts /home, but not in an
# interactive container where /home is read-only -- Inductor's makedirs then fails before the
# first step. Node-local and overridable instead, and set for Inductor as well as Triton:
# the CuTe kernels compile on first use too, so both arms want a cache that survives the run.
KERNEL_CACHE_DIR=${KERNEL_CACHE_DIR:-"/tmp/mcore_cache_$(id -u)"}
mkdir -p "${KERNEL_CACHE_DIR}"
export TRITON_CACHE_DIR="${KERNEL_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${KERNEL_CACHE_DIR}/inductor"
export TMPDIR="${TMPDIR:-${KERNEL_CACHE_DIR}/tmp}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

################################################################
### Parallelism sanity, before burning an allocation
################################################################
# Overridable: a leaked CUDA context on part of a node (zombie process the container's PID 1
# never reaped) can leave fewer usable GPUs than the allocation has.
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NNODES=${SLURM_NNODES:-12}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
MP=$((TP * CP))
# Overridable because the grad-accumulation depth is global batch / DP, and a topology with
# DP=1 would otherwise run 48 micro-batches per step -- fine for training, needlessly slow for
# a profile, which only needs enough steps to reach the profiled one.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-48}
MICRO_BATCH_SIZE=1

if [ $((WORLD_SIZE % MP)) -ne 0 ]; then
    echo "ERROR: world size ${WORLD_SIZE} not divisible by TP*CP = ${MP}" >&2
    exit 1
fi
DP=$((WORLD_SIZE / MP))
if [ $((GLOBAL_BATCH_SIZE % (DP * MICRO_BATCH_SIZE))) -ne 0 ]; then
    echo "ERROR: global batch ${GLOBAL_BATCH_SIZE} not divisible by DP*MBS = $((DP * MICRO_BATCH_SIZE))" >&2
    exit 1
fi
echo "world=${WORLD_SIZE} TP=${TP} CP=${CP} DP=${DP} grad_accum=$((GLOBAL_BATCH_SIZE / (DP * MICRO_BATCH_SIZE)))"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# scontrol is absent inside the training container, so fall back to $0 rather than aborting
# under set -e. Only the env-log filename depends on it.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID:-nojob}_${DATETIME}.env.log
fi
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

################################################################
### Log environment
################################################################
{
echo "<< START PATHS >>"
echo "IMAGE=${IMAGE}"
echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "CP=${CP} CP_LAYOUT=${CP_LAYOUT} TP=${TP} DP=${DP}"
echo "<< END PATHS >>"
echo -e "\n"
echo "<< START GIT >>"
git -C ${MEGATRON_LM_DIR} log --oneline -1
# -uno: scanning untracked files takes over two minutes on this worktree (large .prof_*,
# .a2a_snaps and data-cache directories), which is long enough to matter at launch. Tracked
# modifications are the provenance that reproduces a run.
git -C ${MEGATRON_LM_DIR} status --porcelain --branch -uno
git -C ${MEGATRON_LM_DIR} diff
echo "<< END GIT >>"
echo -e "\n"
echo "<< START ENV >>"
env | grep -v WANDB_API_KEY
echo "<< END ENV >>"
} |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

################################################################
### Schedule
################################################################
# Overridable so a longer-context probe does not need a second copy of this script. Zigzag
# splits the sequence into 2*CP chunks, so SEQ_LEN must stay divisible by 2*CP; the check
# below refuses rather than failing inside the layout conversion after the allocation.
SEQ_LEN=${SEQ_LEN:-131072}
if [ $((SEQ_LEN % (2 * CP))) -ne 0 ]; then
    echo "ERROR: seq length ${SEQ_LEN} not divisible by 2*CP = $((2 * CP))" >&2
    exit 1
fi
# Token budget drives the schedule. TOKEN_BUDGET=0 falls back to ITERS steps (probe mode).
TOKEN_BUDGET=${TOKEN_BUDGET:-0}
if [ "${TOKEN_BUDGET}" -gt 0 ]; then
    TRAIN_SAMPLES=$((TOKEN_BUDGET / SEQ_LEN))
else
    TRAIN_SAMPLES=$((ITERS * GLOBAL_BATCH_SIZE))
fi
# WSD decay spans the whole run; the trailing decay phase keeps the production 15% ratio.
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-15259}
LR_DECAY_SAMPLES=${TRAIN_SAMPLES}
LR_WSD_DECAY_SAMPLES=$((TRAIN_SAMPLES * 15 / 100))

# The production warmup is sized for the 1T-token run and is longer than a short A/B, which
# trips 'assert lr_warmup_steps < lr_decay_steps' in OptimizerParamScheduler -- after argument
# validation, so DRY_RUN cannot see it. Shorten it rather than fail: both arms get the same
# schedule, which is all the comparison requires.
if [ "${LR_WARMUP_SAMPLES}" -ge "${TRAIN_SAMPLES}" ]; then
    LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES / 10))
    echo "note: warmup shortened to ${LR_WARMUP_SAMPLES} samples ($((LR_WARMUP_SAMPLES / GLOBAL_BATCH_SIZE)) steps) for a ${TRAIN_SAMPLES}-sample run"
fi
# Guard the assert at submission time instead of thirty seconds into a 16-node allocation.
if [ $((LR_WARMUP_SAMPLES / GLOBAL_BATCH_SIZE)) -ge $((LR_DECAY_SAMPLES / GLOBAL_BATCH_SIZE)) ]; then
    echo "ERROR: lr_warmup_steps $((LR_WARMUP_SAMPLES / GLOBAL_BATCH_SIZE)) must be < lr_decay_steps $((LR_DECAY_SAMPLES / GLOBAL_BATCH_SIZE))" >&2
    exit 1
fi
echo "schedule: TRAIN_SAMPLES=${TRAIN_SAMPLES} steps=$((TRAIN_SAMPLES / GLOBAL_BATCH_SIZE)) tokens=$((TRAIN_SAMPLES * SEQ_LEN)) warmup_steps=$((LR_WARMUP_SAMPLES / GLOBAL_BATCH_SIZE))"

################################################################
### Context-parallel arguments
################################################################
CP_ARGS=""
if [ "${DRY_RUN:-0}" = "1" ]; then
    # No CP flags at all: DRY_RUN forces a 1x1 layout, and leaving them out keeps the
    # DUMP_CONFIG output free of context-parallel fields that belong to a different PR.
    CP_ARGS=""
elif [ "${CP}" -gt 1 ]; then
    # DSA over GQA asserts allgather: CP all-gathers K and V so each rank runs the indexer
    # against the full key set, which keeps top-k a global selection rather than a
    # distributed merge.
    # Spell it 'all_gather': DSA normalizes via replace("_","") and accepts either, but
    # Transformer Engine string-matches exactly and raises on 'allgather'.
    CP_ARGS="--context-parallel-size ${CP} --cp-comm-type ${CP_COMM_TYPE}"
    if [ "${CP_LAYOUT}" != "zigzag" ]; then
        # Only zigzag is supported under CP. attention_cp_layout='contiguous' is rejected by
        # _validate_cp_layouts, so fail here rather than after the allocation is granted.
        echo "ERROR: CP_LAYOUT='${CP_LAYOUT}' is not supported; only 'zigzag' works with CP." >&2
        exit 1
    fi
fi

OPTIONS=" \
	--dist-ckpt-strictness log_unexpected \
        ${PRETRAINED_ARG} \
        ${CP_ARGS} \
        \
	--experimental-attention-variant dsa \
	--dsa-indexer-n-heads 1 \
	--dsa-indexer-head-dim 128 \
	--dsa-indexer-topk ${TOPK:-512} \
	--dsa-indexer-loss-coeff 0.1 \
	--dsa-indexer-mode simplified \
	${PHASE_ARGS} \
	\
	--dsa-kernel-backend ${DSA_BACKEND} \
	\
	--seed 1234 \
	--num-dataset-builder-threads 4 \
	--rerun-mode disabled \
	\
	--attention-backend flash \
	--num-workers 1 \
	--disable-gloo-process-groups \
	--ckpt-format torch_dist \
	--ckpt-fully-parallel-save \
	--ckpt-fully-parallel-load \
	--ckpt-assume-constant-structure \
	--no-load-rng \
        \
	--squared-relu \
        --no-mmap-bin-files \
	--distributed-timeout-minutes ${DIST_TIMEOUT_MIN:-120} \
	--exit-duration-in-mins ${EXIT_MINS} \
	--no-create-attention-mask-in-dataloader \
        \
        --overlap-grad-reduce \
        --tensor-model-parallel-size ${TP} \
	--pipeline-model-parallel-size 1 \
	--use-distributed-optimizer \
	--override-opt_param-scheduler \
        \
	--mamba-num-heads 128 \
	--mamba-head-dim 64 \
        --mamba-state-dim 128 \
        --mamba-num-groups 8 \
	--untie-embeddings-and-output-weights \
        --init-method-std 0.014 \
        --position-embedding-type none \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 1 \
	--hybrid-layer-pattern M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M- \
	--ffn-hidden-size 21376 \
	--kv-channels 256 \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr-decay-style WSD \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
	--lr-wsd-decay-style minus_sqrt \
	--lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES} \
	--save ${CHECKPOINT_DIR} \
        --load ${CHECKPOINT_DIR} \
	--per-split-data-args-path ${BLEND_PATH} \
	--data-cache-path ${DATACACHE_DIR} \
	--tiktoken-pattern v2 \
	--tokenizer-type TikTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --lr ${PHASE_LR} \
        --min-lr 8.0e-6 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-interval 1 \
	--log-memory-interval 1000 \
	--log-timers-to-tensorboard \
	--log-params-norm \
	--log-num-zeros-in-grad \
	--log-throughput \
	--log-progress \
	--log-energy \
	--logging-level 20 \
        --save-interval ${SAVE_INTERVAL} \
        --eval-interval 1000 \
        --eval-iters 14 \
        --bf16 \
        --use-mcore-models \
        --spec megatron.core.transformer.experimental_attention_variant.dsa_layer_specs dsa_stack_spec \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-exp-name ${WANDB_EXP_NAME} \
        --wandb-save-dir ${WANDB_SAVE_DIR}"

# How each rank's python is started. The default assumes one task per GPU is already in place,
# which is true under the srun below and under the job broker. An interactive `srun --pty bash`
# is a single task, so there is nobody to rendezvous with: set
#   LAUNCHER="python -m torch.distributed.run --nproc-per-node=4 --master-port=29501"
# and torchrun spawns the ranks itself, overriding the session's RANK/WORLD_SIZE/MASTER_PORT.
LAUNCHER=${LAUNCHER:-"python -u"}
RUN_CMD="${LAUNCHER} ${MEGATRON_LM_DIR}/pretrain_hybrid.py ${OPTIONS}"

# PROFILE=1 captures a PyTorch profiler trace instead of just training. PROFILE_ARMS lists the
# kernel backends to run in sequence inside the one allocation, so both arms see the same nodes
# -- at this scale the difference between two allocations is larger than the difference between
# the arms. Each arm needs its own tensorboard directory: the profiler writes its trace to a
# sibling of that directory, so a shared one would have the second arm overwrite the first.
#
# PROFILE_START is later than the single-arm default on purpose. The CuTe kernels compile on
# first use and reserve_indexer runs on the first forward, so the opening iterations of the
# ssa-cute arm are dominated by one-time work that is not what this measures.
if [ "${PROFILE:-0}" = "1" ]; then
    PROFILE_ARMS=${PROFILE_ARMS:-"min-memory-triton ssa-cute"}
    PROFILE_START=${PROFILE_START:-12}
    PROFILE_END=${PROFILE_END:-13}
    PROFILE_CMD=""
    for ARM in ${PROFILE_ARMS}; do
        # Each arm needs its own checkpoint directory as well as its own tensorboard
        # directory. Sharing the checkpoint directory makes the second arm load what the first
        # one saved on its way out, resume at the final iteration and exit with nothing to
        # train -- it looks like a clean run that simply produced no trace.
        #
        # The tensorboard path gets a slash appended before the suffix on purpose: the profiler
        # writes its trace to a *sibling* of that directory, so without the extra component
        # both arms resolve to the same trace file and the second overwrites the first.
        ARM_TB="${TENSORBOARD_DIR%/}_prof_${ARM}/tb"
        ARM_CKPT="${CHECKPOINT_DIR%/}_prof_${ARM}/"
        ARM_OPTIONS=$(echo "${OPTIONS}" \
            | sed "s|--dsa-kernel-backend ${DSA_BACKEND}|--dsa-kernel-backend ${ARM}|" \
            | sed "s|--wandb-exp-name ${WANDB_EXP_NAME}|--wandb-exp-name ${WANDB_EXP_NAME}_${ARM}|" \
            | sed "s|--save ${CHECKPOINT_DIR}|--save ${ARM_CKPT}|" \
            | sed "s|--load ${CHECKPOINT_DIR}|--load ${ARM_CKPT}|" \
            | sed "s|--tensorboard-dir ${TENSORBOARD_DIR}|--tensorboard-dir ${ARM_TB}|")
        if [ "${NSYS:-0}" = "1" ]; then
            # Megatron only calls cudaProfilerStart/Stop at the step boundaries; the capture
            # itself has to be set up on the command line, hence the wrapper. It wraps rank 0
            # alone -- profile_ranks says only rank 0 toggles the API, and sixty-four reports
            # of a few GB each would be unreadable as well as slow.
            NSYS_OUT="${LOGS_DIR}/nsys_${ARM}_${DATETIME}"
            NSYS_CMD="nsys profile -s none -t nvtx,cuda -o ${NSYS_OUT} --force-overwrite true \
                --capture-range=cudaProfilerApi --capture-range-end=stop"
            PY_CMD="${LAUNCHER} ${MEGATRON_LM_DIR}/pretrain_hybrid.py ${ARM_OPTIONS} \
                --profile --nvtx-ranges \
                --profile-step-start ${PROFILE_START} --profile-step-end ${PROFILE_END} \
                --profile-ranks 0"
            ARM_CMD="if [ \"\${SLURM_PROCID:-0}\" = \"0\" ]; then ${NSYS_CMD} ${PY_CMD}; \
                else ${PY_CMD}; fi"
        else
            ARM_CMD="${LAUNCHER} ${MEGATRON_LM_DIR}/pretrain_hybrid.py ${ARM_OPTIONS} \
                --profile --use-pytorch-profiler \
                --profile-step-start ${PROFILE_START} --profile-step-end ${PROFILE_END} \
                --profile-ranks 0"
        fi
        if [ -z "${PROFILE_CMD}" ]; then
            PROFILE_CMD="echo '=== arm ${ARM}'; ${ARM_CMD}"
        else
            PROFILE_CMD="${PROFILE_CMD}; echo '=== arm ${ARM}'; ${ARM_CMD}"
        fi
    done
    RUN_CMD="${PROFILE_CMD}"
fi

# DRY_RUN=1 parses and validates OPTIONS on a single process and exits. Argument validation
# runs before any model build, so a flag the tree no longer defines fails here in seconds
# instead of after a twelve-node allocation is granted. Flags do get deleted: the kernel tile
# sizes, the cache toggles and --dsa-simplified-use-learned-k were all removed by the PR stack.
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN: validating arguments only (1 process, no training)"
    DRY_PY=$(mktemp /tmp/dsa_dry_run_XXXXXX.py)
    cat >"${DRY_PY}" <<'PYEOF'
import sys
sys.argv = ['pretrain_hybrid.py'] + sys.argv[1:]
from megatron.training.arguments import parse_and_validate_args

# Mirror pretrain_hybrid.py's own call, or the check reports failures the real run would not.
try:
    from megatron.post_training.arguments import add_modelopt_args
except ImportError:
    add_modelopt_args = None

parse_and_validate_args(
    extra_args_provider=add_modelopt_args,
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
)
print('DRY_RUN: arguments OK')

# DUMP_CONFIG=1 additionally builds the TransformerConfig this run would use and prints the
# fields that differ from the dataclass defaults. That is the honest answer to "what config
# triggers this": it includes the values __post_init__ resolves, which the flags never show.
import os

if os.environ.get('DUMP_CONFIG') == '1':
    import dataclasses

    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.global_vars import get_args

    config = core_transformer_config_from_args(get_args())
    print('\n--- TransformerConfig fields differing from their defaults ---')
    for field in sorted(dataclasses.fields(config), key=lambda f: f.name):
        default = field.default
        if default is dataclasses.MISSING and field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        value = getattr(config, field.name)
        if default is dataclasses.MISSING or value != default:
            print(f'  {field.name} = {value!r}')
PYEOF
    # Inside the training container there is no srun, and none is needed: a single process
    # is all argument validation requires. Outside, go through srun so the check runs against
    # the same image the real job uses.
    # A failing validation is the expected outcome when a flag has been removed, so do not let
    # errexit kill the script before the status is captured and the temp file cleaned up.
    set +e
    # The helper lives in /tmp, so sys.path[0] is /tmp and the checkout is not importable --
    # and an installed megatron-core would shadow it anyway. Point PYTHONPATH at the worktree
    # so the flags are validated against the code this job will actually run.
    if command -v srun >/dev/null 2>&1; then
        srun -l --nodes=1 --ntasks=1 \
             --mpi=none \
             --container-image=${IMAGE} \
             --container-mounts="/scratch:/scratch,/lustre:/lustre,/tmp:/tmp" \
             --no-container-mount-home \
             --export=ALL \
             sh -c "cd ${MEGATRON_LM_DIR} && PYTHONPATH=${MEGATRON_LM_DIR}:\${PYTHONPATH:-} python -u ${DRY_PY} ${OPTIONS}"
    else
        ( cd "${MEGATRON_LM_DIR}" \
          && PYTHONPATH="${MEGATRON_LM_DIR}:${PYTHONPATH:-}" python -u "${DRY_PY}" ${OPTIONS} )
    fi
    STATUS=$?
    set -e
    rm -f "${DRY_PY}"
    exit ${STATUS}
fi

# NO_SRUN=1 runs the command directly instead of launching its own srun, for when something
# else has already placed this script inside the allocation with one task per GPU -- the job
# broker does that. Launching a second srun from in there would nest allocations.
if [ "${NO_SRUN:-0}" = "1" ]; then
    eval "${RUN_CMD}"
    exit $?
fi

srun -l \
     --mpi=none \
     --container-image=${IMAGE} \
     --container-mounts="/scratch:/scratch,/lustre:/lustre" \
     --no-container-mount-home \
     --export=ALL \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${RUN_CMD}"
