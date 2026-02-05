#!/bin/bash
# GRPO Training Script with TracIn (Influence Function) for Toxicity Reduction
# Based on run_train_iif.sh but using GRPO algorithm
#
# KEY CHANGE: RANDOM validation selection (like PPO)
# - Do NOT filter to high-quality only - need good AND bad for contrast!
# - Use seqloss-lastadv (advantages encode contrast between good/bad)
# - tracin_val_batch_size=64: Use LARGER validation set for stable gradient estimate
#
# Why random validation instead of quality-filtered:
# - TracIn needs CONTRAST between good and bad samples
# - Quality filtering removes bad samples → no contrast → doesn't work!
# - Random selection includes both → advantages encode improvement direction
#
# OPTIMIZED for A100 80GB with hook-based TracIn (memory efficient):
# - batch_size=64: Larger main batch (64 prompts x 8 gens = 512 samples)
# - tracin_batch_size=32: Process 32 samples at a time for gradient computation
# - tracin_val_batch_size=64: Larger validation batch (RANDOM, includes good AND bad!)
# - mini_batch_size=32: Larger training mini-batches
# - val_size=512: Larger validation set pool
# - gen_bsize=128: Larger generation batch for better GPU utilization
# - num_generations=8: SAME AS STANDARD for fair comparison
#
# Note: Hook-based approach reuses computational graphs, preventing memory leaks.

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
    --batch_size=64 \
    --tracin_batch_size=32 \
    --tracin_val_batch_size=64 \
    --mini_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --grpo_epochs=4 \
    --num_generations=8 \
    --seed=22 \
    --max_length=30 \
    --gen_bsize=128 \
    --val_size=512 \
    --learning_rate=1e-5 \
    --early_stopping=False \
    --output_dir=output_tox_grpo_tracin_2.7b_fp16_kl-0.04_val-random-64_gen-8_mbs-32_seed-22 \
    --init_kl_coef=0.04 \
    --steps=1000 \
    --min_length=20 \
    --temperature=1.0 \
    --wandb_project="grpo-detox" \
    --run_name="grpo-tracin-2.7b-fp16-kl-0.04-val-random-64_gen-8_mbs-32_seed-22" \
    --tracin \
    --with_validation \
    --val_loss_type="seqloss-lastadv" \
    --gen_data_dir="gen_tox_grpo_samples_tracin_2.7b_fp16_kl-0.04_val-random-64_gen-8_mbs-32_seed-22"
