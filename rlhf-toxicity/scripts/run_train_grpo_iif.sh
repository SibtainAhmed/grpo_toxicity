#!/bin/bash
# GRPO Training Script with TracIn (Influence Function) for Toxicity Reduction
# Based on run_train_iif.sh but using GRPO algorithm
#
# PERFORMANCE NOTES:
# - mini_batch_size=1 is VERY SLOW (processes 1 sample at a time)
# - Increase mini_batch_size to 8-16 for 8-16x speedup
# - Adjust gradient_accumulation_steps to maintain effective batch size

set -x
accelerate launch --main_process_port=29525 \
    --num_machines 1  \
    --num_processes 1 \
    train_grpo.py --log_with=wandb \
    --model_name="EleutherAI/gpt-neo-2.7B" \
    --dataset_name="toxicity" \
    --reward_model_name="facebook/roberta-hate-speech-dynabench-r4-target" \
    --adafactor=False \
    --save_freq=10 \
    --batch_size=64 \
    --tracin_batch_size=64 \
    --tracin_val_batch_size=64 \
    --mini_batch_size=8 \
    --gradient_accumulation_steps=8 \
    --grpo_epochs=4 \
    --num_generations=4 \
    --seed=22 \
    --max_length=30 \
    --gen_bsize=64 \
    --val_size=1024 \
    --learning_rate=1e-5 \
    --early_stopping=False \
    --output_dir=output_tox_grpo_tracin_2.7b_bfloat16_kl-0.04_val-1024_tgt-seqloss-lastadv_mbs-8_seed-22 \
    --init_kl_coef=0.04 \
    --steps=200 \
    --min_length=20 \
    --temperature=1.0 \
    --wandb_project="grpo-detox" \
    --run_name="grpo-tracin-2.7b-bfloat16-kl-0.04-val-1024_tgt-seqloss-lastadv_mbs-8_seed-22" \
    --tracin \
    --with_validation \
    --val_loss_type="seqloss-lastadv" \
    --gen_data_dir="gen_tox_grpo_samples_tracin_2.7b_bfloat16_kl-0.04_val-1024_tgt-seqloss-lastadv_mbs-8_seed-22"
