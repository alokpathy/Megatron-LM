#!/bin/bash

# Hybrid Transformer + SSM(Mamba) + MoE model with DSA sparse attention.
#
# PER-GPU EMULATION MODE:
#   This script emulates ONE GPU's slice of a large parallel model on a 4-GPU box.
#   You give the *intended* parallelism of the large model via env vars:
#       EMU_TP  tensor-parallel size   (default 1)
#       EMU_EP  expert-parallel size   (default 1)
#       EMU_PP  pipeline-parallel size (default 1)
#   The script divides the large-model GLOBAL dims down to the LOCAL per-GPU dims
#   and runs a pure 4-way data-parallel job (real TP=EP=PP=1) with those local
#   shapes. Each of the 4 GPUs then sees the same DSA/attention/MLP/MoE workload
#   a single GPU would see in the full large run.
#
#   TP splits: attention query heads, ffn, moe-ffn, shared-expert, mamba heads/groups.
#   EP splits: number of experts (router top-k is clamped to local expert count).
#   PP splits: number of layers.
#   Unchanged per-GPU: hidden, seq-len, attn head dim, kv heads(=1, replicated),
#   mamba head/state dim, and the DSA indexer dims (indexer weights are TP-duplicated).
#
# Use: EMU_TP=8 EMU_EP=64 EMU_PP=4 bash train_hybrid_moe.sh [run-name] [1=dsa] [1=nsys] [backend] [seq-len] [1=wandb]
# backend: triton = triton-min-memory, torch = torch-min-memory, cudnn = triton-min-memory + cuDNN indexer

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0

export TRITON_CACHE_DIR="./triton-cache/"

ROOT_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/atripathy/Megatron-LM"
NAME="${1:-hybrid_transformer_ssm_moe}"
USE_DSA="${2:-1}"
USE_NSYS="${3:-0}"
DSA_BACKEND="${4:-triton}"  # triton | torch | cudnn
SEQ_LEN="${5:-8192}"
USE_WANDB="${6:-0}"

# --- Emulated parallelism of the target large model ---
EMU_TP="${EMU_TP:-1}"
EMU_EP="${EMU_EP:-1}"
EMU_PP="${EMU_PP:-1}"

# --- DSA min-memory query chunk size (unset -> defaults to min(seq_len, 512)) ---
QBLOCK="${QBLOCK:-}"
if [ -n "${QBLOCK}" ]; then
    QBLOCK_ARG="--dsa-kernel-query-block-size ${QBLOCK}"
else
    QBLOCK_ARG=""
fi

echo "=================== script inputs ==================="
echo "positional: NAME=${NAME} USE_DSA=${USE_DSA} USE_NSYS=${USE_NSYS} DSA_BACKEND=${DSA_BACKEND} SEQ_LEN=${SEQ_LEN} USE_WANDB=${USE_WANDB}"
echo "env:        EMU_TP=${EMU_TP} EMU_EP=${EMU_EP} EMU_PP=${EMU_PP} TRAIN_ITERS=${TRAIN_ITERS:-<unset>} QBLOCK=${QBLOCK:-<default>}"
echo "raw argv:   $0 $@"
echo "====================================================="

if [ "${DSA_BACKEND}" = "triton" ]; then
    DSA_KERNEL_BACKEND="triton-min-memory"; DSA_EXTRA=""
elif [ "${DSA_BACKEND}" = "torch" ]; then
    DSA_KERNEL_BACKEND="torch-min-memory"; DSA_EXTRA=""
elif [ "${DSA_BACKEND}" = "cudnn" ]; then
    DSA_KERNEL_BACKEND="triton-min-memory"; DSA_EXTRA="--dsa-use-cudnn"
else
    echo "Unknown DSA_BACKEND=${DSA_BACKEND}. Use triton, torch, or cudnn." && exit 1
fi

# ================= GLOBAL (target large-model) dimensions =================
G_HIDDEN=10240
G_VOCAB=262144
G_HEADS=80            # attention query heads   (TP-split)
G_KV_HEADS=1          # num_query_groups        (replicated across TP -> unchanged)
G_ATTN_HEAD_DIM=256   # kv_channels             (unchanged)
G_FFN=10240           # dense ffn (fallback)    (TP-split)
G_MOE_FFN=10240       # moe ffn hidden          (TP-split)
G_MOE_SHARED=12288    # shared expert size      (TP-split)
G_EXPERTS=512         # routed experts          (EP-split)
G_ROUTER_TOPK=10      # router top-k            (clamped to local experts)
G_MOE_LATENT=3072     # moe latent size         (unchanged; DSA-orthogonal)
G_MAMBA_HEADS=80      # (TP-split)
G_MAMBA_GROUPS=80     # (TP-split)
G_MAMBA_HEAD_DIM=64   # (unchanged)
G_MAMBA_STATE=128     # (unchanged)
G_IDX_HEADS=64        # DSA indexer heads       (TP-duplicated -> unchanged)
G_IDX_HEAD_DIM=128    # DSA indexer head dim    (unchanged)
G_IDX_TOPK=2048       # DSA indexer topk        (unchanged)
G_N_ATTN=12           # attention layers        (PP-split)
G_N_SSM=70            # mamba layers            (PP-split)
G_N_MOE=70            # moe layers              (PP-split)

# ================= divide GLOBAL -> LOCAL (per-GPU) =================
div() { # value divisor name  -> prints value/divisor, exits if not divisible
    if [ $(( $1 % $2 )) -ne 0 ]; then
        echo "ERROR: cannot emulate: ${3}=${1} not divisible by ${2}." >&2; exit 1
    fi
    echo $(( $1 / $2 ))
}

L_HIDDEN=${G_HIDDEN}
L_VOCAB=${G_VOCAB}
L_KV_HEADS=${G_KV_HEADS}
L_ATTN_HEAD_DIM=${G_ATTN_HEAD_DIM}
L_MAMBA_HEAD_DIM=${G_MAMBA_HEAD_DIM}
L_MAMBA_STATE=${G_MAMBA_STATE}
L_MOE_LATENT=${G_MOE_LATENT}
L_IDX_HEADS=${G_IDX_HEADS}
L_IDX_HEAD_DIM=${G_IDX_HEAD_DIM}
L_IDX_TOPK=${G_IDX_TOPK}

L_HEADS=$(div ${G_HEADS} ${EMU_TP} "num_attention_heads") || exit 1
L_FFN=$(div ${G_FFN} ${EMU_TP} "ffn_hidden_size") || exit 1
L_MOE_FFN=$(div ${G_MOE_FFN} ${EMU_TP} "moe_ffn_hidden_size") || exit 1
L_MOE_SHARED=$(div ${G_MOE_SHARED} ${EMU_TP} "moe_shared_expert_intermediate_size") || exit 1
L_MAMBA_HEADS=$(div ${G_MAMBA_HEADS} ${EMU_TP} "mamba_num_heads") || exit 1
L_MAMBA_GROUPS=$(div ${G_MAMBA_GROUPS} ${EMU_TP} "mamba_num_groups") || exit 1
L_EXPERTS=$(div ${G_EXPERTS} ${EMU_EP} "num_experts") || exit 1

# Router top-k is global (routes to top-k of ALL experts); with EP-emulation each
# GPU hosts only local experts, so clamp top-k to what is locally available.
L_ROUTER_TOPK=${G_ROUTER_TOPK}
if [ "${L_ROUTER_TOPK}" -gt "${L_EXPERTS}" ]; then
    echo "WARN: router top-k ${G_ROUTER_TOPK} > local experts ${L_EXPERTS}; clamping to ${L_EXPERTS}. (EP emulation reproduces per-GPU expert COUNT, not global routing.)"
    L_ROUTER_TOPK=${L_EXPERTS}
fi

# Megatron requires pre-softmax routing when top-k == 1 (happens when EP-emulation
# leaves a single local expert, e.g. EMU_EP=512). Post-softmax over 1 expert is a
# no-op it refuses to run; pre-softmax is well-defined there.
PRESOFTMAX_ARG=""
if [ "${L_ROUTER_TOPK}" -eq 1 ]; then
    PRESOFTMAX_ARG="--moe-router-pre-softmax"
fi

# PP splits layers (floor division; representative interleaved mix regenerated below)
L_N_ATTN=$(( G_N_ATTN / EMU_PP ))
L_N_SSM=$(( G_N_SSM / EMU_PP ))
L_N_MOE=$(( G_N_MOE / EMU_PP ))

# Build the hybrid layer pattern from local counts (M=Mamba, *=attention, E=MoE).
HYBRID_LAYER_PATTERN=$(python3 - "${L_N_ATTN}" "${L_N_SSM}" "${L_N_MOE}" <<'PYEOF'
import sys
n_attn, n_ssm, n_moe = (int(x) for x in sys.argv[1:4])
base = [('M' if k % 2 == 0 else 'E') for k in range(n_ssm + n_moe)]
# guarantee exact per-type counts: base has ceil half M / floor half E; fix up
base = ['M'] * n_ssm + ['E'] * n_moe
# interleave M/E for a representative mix
mix = []
i, j = 0, n_ssm
for k in range(n_ssm + n_moe):
    if k % 2 == 0 and i < n_ssm:
        mix.append('M'); i += 1
    elif j < n_ssm + n_moe:
        mix.append('E'); j += 1
    else:
        mix.append('M'); i += 1
# splice in exactly n_attn attention layers at even intervals
chunks = n_attn + 1
out, idx = [], 0
L = len(mix)
for c in range(chunks):
    end = idx + L // chunks + (1 if c < L % chunks else 0)
    out.extend(mix[idx:end])
    if c < n_attn:
        out.append('*')
    idx = end
s = ''.join(out)
assert s.count('M') == n_ssm and s.count('E') == n_moe and s.count('*') == n_attn, \
    (s.count('M'), s.count('E'), s.count('*'), n_ssm, n_moe, n_attn)
print(s)
PYEOF
) || { echo "ERROR: failed to build hybrid pattern (check local layer counts)"; exit 1; }

echo "=================== emulation summary ==================="
echo "EMU_TP=${EMU_TP} EMU_EP=${EMU_EP} EMU_PP=${EMU_PP}  (target large-model parallelism)"
echo "attn heads:  ${G_HEADS} -> ${L_HEADS}   ffn: ${G_FFN} -> ${L_FFN}   moe-ffn: ${G_MOE_FFN} -> ${L_MOE_FFN}"
echo "shared-exp:  ${G_MOE_SHARED} -> ${L_MOE_SHARED}   experts: ${G_EXPERTS} -> ${L_EXPERTS}   router-topk: ${G_ROUTER_TOPK} -> ${L_ROUTER_TOPK}"
echo "mamba heads/groups: ${G_MAMBA_HEADS}/${G_MAMBA_GROUPS} -> ${L_MAMBA_HEADS}/${L_MAMBA_GROUPS}"
echo "layers attn/ssm/moe: ${G_N_ATTN}/${G_N_SSM}/${G_N_MOE} -> ${L_N_ATTN}/${L_N_SSM}/${L_N_MOE}  (total $(( L_N_ATTN + L_N_SSM + L_N_MOE )))"
echo "DSA indexer (unchanged): heads=${L_IDX_HEADS} head_dim=${L_IDX_HEAD_DIM} topk=${L_IDX_TOPK}"
echo "NAME=${NAME} USE_DSA=${USE_DSA} USE_NSYS=${USE_NSYS} DSA_BACKEND=${DSA_BACKEND} SEQ_LEN=${SEQ_LEN} USE_WANDB=${USE_WANDB}"
echo "pattern: ${HYBRID_LAYER_PATTERN}"
echo "========================================================"

TOKENIZER_MODEL="${ROOT_DIR}/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"
BLEND_PATH="${ROOT_DIR}/blend_files/1t_singlephase.json"

CHECKPOINT_DIR="${ROOT_DIR}/${NAME}/checkpoints"
DATACACHE_DIR="${ROOT_DIR}/data_cache"
TENSORBOARD_DIR="${ROOT_DIR}/tensorboard/${NAME}"
PROFILE_DIR="${ROOT_DIR}/profiles"
mkdir -p ${CHECKPOINT_DIR} ${DATACACHE_DIR} ${TENSORBOARD_DIR} ${PROFILE_DIR}

cd ${ROOT_DIR}
export PYTHONPATH=${ROOT_DIR}:$PYTHONPATH

GBS=4
if [ -n "${TRAIN_ITERS}" ]; then
    TRAIN_SAMPLES=$((TRAIN_ITERS * GBS))
else
    TRAIN_SAMPLES=122070313
fi
TRAIN_ITERS_ARGS="--train-samples ${TRAIN_SAMPLES} \
    --lr-warmup-samples 3051758 \
    --lr-decay-samples 122070313 \
    --lr-wsd-decay-samples 24414063"

$( [ "${USE_NSYS}" = "1" ] && echo "nsys profile \
    -s none \
    -t nvtx,cuda \
    -o ${ROOT_DIR}/profiles/${NAME} \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop" ) \
torchrun \
    --nproc-per-node 4 \
    pretrain_mamba.py \
    --use-mcore-models \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --hybrid-layer-pattern ${HYBRID_LAYER_PATTERN} \
    --hidden-size ${L_HIDDEN} \
    --ffn-hidden-size ${L_FFN} \
    --padded-vocab-size ${L_VOCAB} \
    --num-attention-heads ${L_HEADS} \
    --group-query-attention \
    --num-query-groups ${L_KV_HEADS} \
    --kv-channels ${L_ATTN_HEAD_DIM} \
    --mamba-num-heads ${L_MAMBA_HEADS} \
    --mamba-num-groups ${L_MAMBA_GROUPS} \
    --mamba-head-dim ${L_MAMBA_HEAD_DIM} \
    --mamba-state-dim ${L_MAMBA_STATE} \
    --num-experts ${L_EXPERTS} \
    --moe-ffn-hidden-size ${L_MOE_FFN} \
    --moe-shared-expert-intermediate-size ${L_MOE_SHARED} \
    --moe-router-topk ${L_ROUTER_TOPK} \
    ${PRESOFTMAX_ARG} \
    --moe-latent-size ${L_MOE_LATENT} \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --squared-relu \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.014 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --rotary-percent 1.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    --bf16 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    ${TRAIN_ITERS_ARGS} \
    --lr-decay-style WSD \
    --micro-batch-size 1 \
    --global-batch-size ${GBS} \
    --lr 8e-4 \
    --min-lr 8e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 1000 \
    --eval-iters 14 \
    --per-split-data-args-path ${BLEND_PATH} \
    --data-cache-path ${DATACACHE_DIR} \
    --tokenizer-type TikTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tiktoken-pattern v2 \
    --no-mmap-bin-files \
    --num-workers 1 \
    --no-create-attention-mask-in-dataloader \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --tensor-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --ddp-num-buckets 8 \
    --attention-backend flash \
    --ckpt-format torch_dist \
    --load ${CHECKPOINT_DIR} \
    --save ${CHECKPOINT_DIR} \
    --save-interval 500 \
    --save-retain-interval 2000 \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --async-save \
    --use-persistent-ckpt-worker \
    --ckpt-assume-constant-structure \
    $( [ "${USE_WANDB}" = "1" ] && echo "\
    --wandb-project atripathy-cudnn-dsa \
    --wandb-exp-name ${NAME}" ) \
    --log-interval 1 \
    --log-memory-interval 1 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-progress \
    --log-energy \
    --logging-level 20 \
    --timing-log-option minmax \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    --manual-gc \
    --manual-gc-interval 10 \
    --distributed-timeout-minutes 10 \
    --exit-duration-in-mins 235 \
    --disable-gloo-process-groups \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    $( [ "${USE_NSYS}" = "1" ] && echo "\
    --profile \
    --profile-step-start 5 \
    --profile-step-end 8 \
    --profile-ranks 0" ) \
    $( [ "${USE_DSA}" = "1" ] && echo "\
    --experimental-attention-variant dsa \
    --dsa-kernel-backend ${DSA_KERNEL_BACKEND} \
    --dsa-indexer-n-heads ${L_IDX_HEADS} \
    --dsa-indexer-head-dim ${L_IDX_HEAD_DIM} \
    --dsa-indexer-topk ${L_IDX_TOPK} \
    --dsa-indexer-use-hadamard \
    --dsa-indexer-loss-coeff 0.01 \
    --dsa-min-memory-profile \
    --dsa-min-memory-profile-rank 0 \
    --dsa-kernel-cache-routing \
    --dsa-kernel-cache-indexer-k \
    --dsa-kernel-cache-selected-scores \
    --dsa-indexer-use-sparse-loss \
    ${QBLOCK_ARG} \
    ${DSA_EXTRA} \
    --no-rope-fusion" )
