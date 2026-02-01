#!/bin/bash
# GRPO Training Script with TracIn (Influence Function) for Toxicity Reduction
# Based on run_train_iif.sh but using GRPO algorithm
#
# MEMORY OPTIMIZATIONS for TracIn (v3 - for 80GB GPU near capacity):
# - batch_size=16: Very small main batch (16 prompts x 4 gens = 64 samples)
# - tracin_batch_size=4: Process 4 samples at a time for gradient computation
# - tracin_val_batch_size=4: Small validation batch for gradient computation
# - mini_batch_size=4: Small training mini-batches
# - val_size=256: Smaller validation set
# - gen_bsize=16: Smaller generation batch
# - Memory cleanup added between steps
#
# Note: GPT-Neo-2.7B + LoRA + reward model uses ~70GB baseline.
# TracIn adds gradient overhead. These settings are tuned for 80GB GPUs.

set -x
accelerate launch --main_process_port=29525 \
    --num_machines 1  \
    --num_processes 1 \
    --mixed_precision fp16 \
    train_grpo.py --log_with=wandb \
    --model_name="EleutherAI/gpt-neo-2.7B" \
    --dataset_name="toxicity" \
    --reward_model_name="facebook/roberta-hate-speech-dynabench-r4-target" \
    --adafactor=False \
    --save_freq=10 \
    --batch_size=16 \
    --tracin_batch_size=4 \
    --tracin_val_batch_size=4 \
    --mini_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --grpo_epochs=4 \
    --num_generations=4 \
    --seed=22 \
    --max_length=30 \
    --gen_bsize=16 \
    --val_size=256 \
    --learning_rate=1e-5 \
    --early_stopping=False \
    --output_dir=output_tox_grpo_tracin_2.7b_fp16_kl-0.04_val-256_tgt-seqloss-lastadv_mbs-4_seed-22 \
    --init_kl_coef=0.04 \
    --steps=200 \
    --min_length=20 \
    --temperature=1.0 \
    --wandb_project="grpo-detox" \
    --run_name="grpo-tracin-2.7b-fp16-kl-0.04-val-256_tgt-seqloss-lastadv_mbs-4_seed-22" \
    --tracin \
    --with_validation \
    --val_loss_type="seqloss-lastadv" \
    --gen_data_dir="gen_tox_grpo_samples_tracin_2.7b_fp16_kl-0.04_val-256_tgt-seqloss-lastadv_mbs-4_seed-22"
