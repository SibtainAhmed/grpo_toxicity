#!/bin/bash
# GRPO Standard Training Script for Toxicity Reduction
# Based on run_train_std.sh but using GRPO algorithm

set -x
accelerate launch --main_process_port=29523 \
    --num_machines 1  \
    --num_processes 1 \
    train_grpo.py --log_with=wandb \
    --model_name="EleutherAI/gpt-neo-2.7B" \
    --dataset_name="toxicity" \
    --reward_model_name="facebook/roberta-hate-speech-dynabench-r4-target" \
    --adafactor=False \
    --save_freq=10 \
    --batch_size=64 \
    --mini_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --grpo_epochs=4 \
    --num_generations=4 \
    --seed=22 \
    --max_length=30 \
    --gen_bsize=64 \
    --val_size=1024 \
    --learning_rate=1e-5 \
    --early_stopping=False \
    --output_dir=output_tox_grpo_std_2.7b_bfloat16_kl-0.04_mbs-1_seed-22 \
    --init_kl_coef=0.04 \
    --steps=200 \
    --min_length=20 \
    --temperature=1.0 \
    --wandb_project="grpo-detox" \
    --run_name="grpo-std-2.7b-bfloat16_kl-0.04_mbs-1_seed-22" \
    --gen_data_dir="gen_tox_grpo_samples_std_2.7b_bfloat16_kl-0.04_mbs-1_seed-22" \
