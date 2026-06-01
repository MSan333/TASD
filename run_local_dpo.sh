#!/bin/bash
# =============================================================================
# Local DPO Training Script
#
# Usage:
#   bash run_local_dpo.sh [--lora]
#
# Examples:
#   # Full fine-tune with 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_local_dpo.sh
#
#   # LoRA fine-tune with 2 GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash run_local_dpo.sh --lora
# =============================================================================

set -e

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PYTHONPATH"

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-datasets/dpo}"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/test.parquet"

BETA="${BETA:-0.1}"
LOSS_TYPE="${LOSS_TYPE:-sigmoid}"
LR="${LR:-5e-7}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

LORA_RANK=0
for arg in "$@"; do
    case "$arg" in
        --lora) LORA_RANK=32 ;;
    esac
done

# ── Detect GPUs ───────────────────────────────────────────────────────────────
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
else
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
fi

echo "============================================================"
echo "DPO Training (Local)"
echo "  Model     : $MODEL_PATH"
echo "  Data      : $DATA_DIR"
echo "  GPUs      : $N_GPUS"
echo "  Beta      : $BETA"
echo "  Loss      : $LOSS_TYPE"
echo "  LR        : $LR"
echo "  LoRA rank : $LORA_RANK"
echo "  Batch size: $TRAIN_BATCH_SIZE"
echo "============================================================"

# ── Run ───────────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=$N_GPUS -m verl.trainer.fsdp_dpo_trainer \
    --config-name dpo_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    algorithm.beta=$BETA \
    algorithm.loss_type=$LOSS_TYPE \
    model.partial_pretrain="$MODEL_PATH" \
    model.lora_rank=$LORA_RANK \
    optim.lr=$LR \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.project_name="DPO-local" \
    trainer.experiment_name="dpo-${LOSS_TYPE}-lr${LR}-beta${BETA}" \
    "trainer.logger=[console]"
