#!/bin/bash
source /opt/conda/bin/activate python3.10
cd /home/guoshuaile.gsl/code/TASD

export DATASET=ranking
export MODEL_PATH=/home/guoshuaile.gsl/models/qwen3-8b
export LR=5e-7
export TRAIN_BATCH_SIZE=16
export MINI_BATCH_SIZE=16
export ROLLOUT_N=8
export JOB_NAME=grpo-ranking-4gpu

bash nebula_scripts/grpo/grpo_ranking_gsl.sh
