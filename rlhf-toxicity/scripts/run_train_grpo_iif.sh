#!/bin/bash
# GRPO Training Script with TracIn (Influence Function) for Toxicity Reduction
#
# KEY CHANGES (GROUP-LEVEL TracIn):
# 1. GROUP-LEVEL SELECTION: In GRPO, each prompt produces num_generations
#    responses forming a contrastive group (good vs bad). Per-sample selection
#    breaks this structure. We now select/reject ENTIRE prompt groups based on
#    the sum of per-sample influence scores within the group.
#
# 2. MATCHING VALIDATION LOSS: PPO step_part_I uses the SAME loss formulation
#    (token-level, same advantages) for validation as training. We now do too:
#    val_loss = -mean(advantages_expanded * logprobs * masks)  (rough-orig)
#    using the SAME group-level advantages as training.
#
# Same-batch TracIn (like PPO step_part_I): training batch = validation batch.
#
# OPTIMIZED for A100 80GB with hook-based TracIn (memory efficient):
# - batch_size=64: 8 prompts x 8 gens = 64 samples
# - tracin_batch_size=32: Process 32 samples at a time for gradient computation
# - mini_batch_size=32: Larger training mini-batches
# - gen_bsize=128: Larger generation batch for better GPU utilization
# - num_generations=8: SAME AS STANDARD for fair comparison

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
    --output_dir=output_tox_grpo_tracin_2.7b_fp16_kl-0.04_samebatch_gen-8_mbs-32_seed-22 \
    --init_kl_coef=0.04 \
    --steps=1000 \
    --min_length=20 \
    --temperature=1.0 \
    --wandb_project="grpo-detox" \
    --run_name="grpo-tracin-2.7b-fp16-kl-0.04-samebatch_gen-8_mbs-32_seed-22" \
    --tracin \
    --with_validation \
    --val_loss_type="rough-orig" \
    --gen_data_dir="gen_tox_grpo_samples_tracin_2.7b_fp16_kl-0.04_samebatch_gen-8_mbs-32_seed-22"
