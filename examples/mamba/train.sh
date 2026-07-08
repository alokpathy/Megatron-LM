#!/bin/bash

# Use: bash train.sh [run-name] [1=enable-dsa (default 1)] [1=enable-nsys (default 0)] [backend (default triton)] [seq-len (default 8192)] [1=enable-wandb (default 0)]
# backend: triton = triton-min-memory, torch = torch-min-memory, cudnn = triton-min-memory + cuDNN indexer
# e.g. bash train.sh my_run 1 0 cudnn 16384 1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0

export TRITON_CACHE_DIR="./triton-cache/"

ROOT_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/atripathy/Megatron-LM"
NAME="${1:-8b_hybrid_dsa}"
USE_DSA="${2:-1}"
USE_NSYS="${3:-0}"
DSA_BACKEND="${4:-triton}"  # triton | torch | cudnn
SEQ_LEN="${5:-8192}"
USE_WANDB="${6:-0}"

if [ "${DSA_BACKEND}" = "triton" ]; then
    DSA_KERNEL_BACKEND="triton-min-memory"
    DSA_EXTRA=""
elif [ "${DSA_BACKEND}" = "torch" ]; then
    DSA_KERNEL_BACKEND="torch-min-memory"
    DSA_EXTRA=""
elif [ "${DSA_BACKEND}" = "cudnn" ]; then
    DSA_KERNEL_BACKEND="triton-min-memory"
    DSA_EXTRA="--dsa-use-cudnn"
else
    echo "Unknown DSA_BACKEND=${DSA_BACKEND}. Use triton, torch, or cudnn." && exit 1
fi

echo "NAME=${NAME} USE_DSA=${USE_DSA} USE_NSYS=${USE_NSYS} DSA_BACKEND=${DSA_BACKEND} SEQ_LEN=${SEQ_LEN} USE_WANDB=${USE_WANDB}"

TOKENIZER_MODEL="${ROOT_DIR}/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"
BLEND_PATH="${ROOT_DIR}/blend_files/1t_singlephase.json"

CHECKPOINT_DIR="${ROOT_DIR}/${NAME}/checkpoints"
DATACACHE_DIR="${ROOT_DIR}/data_cache"
TENSORBOARD_DIR="${ROOT_DIR}/tensorboard/${NAME}"
PROFILE_DIR="${ROOT_DIR}/profiles"
mkdir -p ${CHECKPOINT_DIR} ${DATACACHE_DIR} ${TENSORBOARD_DIR} ${PROFILE_DIR}

cd ${ROOT_DIR}
export PYTHONPATH=${ROOT_DIR}:$PYTHONPATH

# 8B Nemotron-H hybrid: 60 layers, 4 attention + Mamba + MLP
HYBRID_LAYER_PATTERN="M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-"

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
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --ffn-hidden-size 21504 \
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
    --global-batch-size 4 \
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
    --tensor-model-parallel-size 2 \
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
    --dsa-indexer-n-heads 32 \
    --dsa-indexer-head-dim 64 \
    --dsa-indexer-topk 1024 \
    --dsa-indexer-use-hadamard \
    --dsa-indexer-loss-coeff 0.01 \
    --dsa-min-memory-profile \
    --dsa-min-memory-profile-rank 0 \
    --dsa-kernel-cache-indexer-k \
    --dsa-indexer-use-sparse-loss \
    ${DSA_EXTRA} \
    --no-rope-fusion" )
