# GRPO Training Script for Toxicity Reduction
# Based on train_rlhf.py but adapted for Group Relative Policy Optimization

import os
import sys
import time
import gc
import torch
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import wandb
from statistics import mean, stdev

# Add scripts directory to path for local imports
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grpo_trainer import GRPOTrainer
from grpo_config import GRPOConfig

from rlhfutils.data import (
    build_toxicity_promptdata,
    collator,
    anscat,
)

# ============================================================================
# GRPOScriptArguments - Defined locally to avoid import issues
# ============================================================================
@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO (Group Relative Policy Optimization) training.
    """
    
    min_length: Optional[int] = field(default=20, metadata={"help": "minimum length for generation"})
    model_name: Optional[str] = field(default="facebook/opt-125m", metadata={"help": "the model name"})
    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the adapter model name"})
    reward_model_name: Optional[str] = field(default="function:bagofwords", metadata={"help": "the reward model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="ultra", metadata={"help": "the dataset name"})
    dataset_path: Optional[str] = field(default=None, metadata={"help": "the dataset path"})
    val_data_path: Optional[str] = field(default='toxicity.csv', metadata={"help": "the validation dataset path"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    kl_penalty: Optional[str] = field(default="kl", metadata={"help": "kl penalty setup"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=256, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the GRPO minibatch size"})
    tracin_batch_size: Optional[int] = field(default=4, metadata={"help": "Number of samples used for TRACIN"})
    tracin_val_batch_size: Optional[int] = field(default=4, metadata={"help": "Number of samples used for TRACIN validation"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    val_size: Optional[int] = field(default=2048, metadata={"help": "the validation size"})
    
    # GRPO-specific: number of generations per prompt
    num_generations: Optional[int] = field(default=4, metadata={"help": "number of generations per prompt for group-relative advantages"})
    grpo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of GRPO epochs"})
    
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    save_rollouts: Optional[bool] = field(default=False, metadata={"help": "save rollouts, rewards to file"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    tracin: Optional[bool] = field(default=False, metadata={"help": "whether to use tracin for reselection"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "whether to load in 8 bit"})
    with_validation: Optional[bool] = field(default=False, metadata={"help": "whether to use validation"})
    reward_source: Optional[str] = field(default="rm", metadata={"help": "the reward source"})
    val_loss_type: Optional[str] = field(default='rough-orig', metadata={"help": "the validation loss type"})
    reward_baseline: Optional[float] = field(default=0, metadata={"help": "a baseline value that is subtracted from the reward"})
    omit_long: Optional[float] = field(default=0, metadata={"help": "whether to omit outputs that don't fit in length context or not"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "length penalty"})
    scale_reward: Optional[float] = field(default=0, metadata={"help": "scale reward"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="checkpoints/debugging", metadata={"help": "directory to save the model"})
    seed: Optional[int] = field(default=1, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=10000, metadata={"help": "number of steps"})
    init_kl_coef: Optional[float] = field(default=0.2, metadata={"help": "Initial KL penalty coefficient"})
    gen_bsize: Optional[int] = field(default=1, metadata={"help": "generation batch size"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "sampling temperature for generation"})
    wandb_project: Optional[str] = field(default="llamatrl", metadata={"help": "wandb project name"})
    run_name: Optional[str] = field(default="llamatrl", metadata={"help": "wandb run name"})
    gen_data_dir: Optional[str] = field(default=None, metadata={"help": "directory to save generated data"})
    val_strategy: Optional[str] = field(default="random", metadata={"help": "validation strategy"})


# ============================================================================
# LoRA config and model loading - Defined locally to avoid import issues
# ============================================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def get_reward_pipeline(rmname, device):
    """Load reward model and tokenizer."""
    if "hate" in rmname:
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
        toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
        toxicity_model = RobertaForSequenceClassification.from_pretrained(
            toxicity_model_id,
        ).to(device)
        return toxicity_tokenizer, toxicity_model
    else:
        raise ValueError(f"Reward model {rmname} not supported for GRPO toxicity training")


def load_models_grpo(script_args, loadms="rmgrpo"):
    """
    Load models for GRPO training.
    Unlike PPO, GRPO uses AutoModelForCausalLM (no value head).
    """
    current_device = Accelerator().local_process_index

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        print("resetting pad token?")
        tokenizer.pad_token = tokenizer.eos_token

    if "grpo" in loadms:
        config = GRPOConfig(
            model_name=script_args.model_name,
            learning_rate=script_args.learning_rate,
            log_with="wandb",
            batch_size=script_args.batch_size,
            val_size=script_args.val_size,
            mini_batch_size=script_args.mini_batch_size,
            tracin_batch_size=script_args.tracin_batch_size,
            tracin_val_batch_size=script_args.tracin_val_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            early_stopping=script_args.early_stopping,
            target_kl=script_args.target_kl,
            grpo_epochs=script_args.grpo_epochs,
            num_generations=script_args.num_generations,
            seed=script_args.seed,
            cliprange=0.2,
            horizon=10000,
            target=script_args.target_kl,
            init_kl_coef=script_args.init_kl_coef,
            steps=script_args.steps,
            kl_penalty=script_args.kl_penalty,
            remove_unused_columns=False,
            val_loss_type=script_args.val_loss_type,
            temperature=script_args.temperature,
        )
        
        # Load model WITHOUT value head (key difference from PPO)
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            load_in_8bit=True if script_args.load_in_8bit else False,
            device_map={"": current_device},
            torch_dtype=torch.bfloat16,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        optimizer = None
        if script_args.adafactor:
            from transformers import Adafactor
            optimizer = Adafactor(
                filter(lambda p: p.requires_grad, model.parameters()),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=config.learning_rate,
            )

    if "rm" in loadms:
        ptok, reward_model = get_reward_pipeline(script_args.reward_model_name, current_device)
    if loadms == "rm":
        return ptok, reward_model
    
    model.gradient_checkpointing_disable()
    
    # GRPO model only
    if loadms == "grpo":
        return config, tokenizer, model, optimizer
    # GRPO with reward model
    return config, tokenizer, model, optimizer, reward_model, ptok

os.environ["WANDB_TAGS"] = "[\"llamatrl\", \"grpo\"]"
tqdm.pandas()

def set_seed(seed):
    """Set seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_reward_scores(reward_model, reward_tokenizer, texts, device):
    """
    Get reward scores from the toxicity classifier.
    Returns negative toxicity (higher = less toxic = better).
    
    IMPORTANT: Uses raw logits (like PPO) not softmax probabilities.
    This gives the same reward scale as PPO (~2 to ~5 range).
    """
    if reward_model is None:
        return [0.0] * len(texts)
    
    # Tokenize
    inputs = reward_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # Use RAW LOGITS like PPO does (not softmax probabilities!)
        # logits[:, 0] is the "nothate" logit - higher = less toxic
        # This gives rewards in range ~2 to ~5, matching PPO scale
        logits = outputs.logits.float()
        scores = logits[:, 0]  # "nothate" logit
        scores_list = scores.cpu().tolist()
    
    # FREE reward model intermediate tensors
    del inputs, outputs, logits, scores
    
    return scores_list


def grpo_train_loop(
    script_args,
    grpo_trainer,
    reward_model,
    tokenizer,
    reward_tokenizer,
    min_length=20,
):
    """
    Standard GRPO training loop without TracIn.
    
    Key difference from PPO: 
    - Generate num_generations responses per prompt
    - Compute group-relative advantages
    """
    print("=" * 50)
    print("Starting GRPO Training Loop")
    print(f"Steps: {script_args.steps}")
    print(f"Batch size: {script_args.batch_size}")
    print(f"Num generations per prompt: {script_args.num_generations}")
    print("=" * 50)
    
    # Generation kwargs
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": script_args.max_length - min_length,
        "temperature": script_args.temperature,
    }
    
    dataloader = grpo_trainer.dataloader
    device = grpo_trainer.current_device
    
    epoch = 0
    for batch in tqdm(dataloader, desc="GRPO Training"):
        if epoch >= script_args.steps:
            break
        epoch += 1
        
        timing = {}
        t0 = time.time()
        
        # Get query tensors
        question_tensors = batch["input_ids"]
        queries_text = batch.get("query", tokenizer.batch_decode(question_tensors, skip_special_tokens=True))
        
        if epoch == 1:
            print("\n=== GRPO Input Sample ===")
            print(queries_text[:2] if len(queries_text) >= 2 else queries_text)
            print("=" * 30)
        
        # Generate multiple responses per query using BATCHED generation
        t = time.time()
        all_queries = []
        
        # Expand queries: repeat each query num_generations times
        expanded_queries = []
        for q_tensor in question_tensors:
            for _ in range(script_args.num_generations):
                expanded_queries.append(q_tensor.to(device))
                all_queries.append(q_tensor)
        
        # Batched generation - process all at once (much faster!)
        with torch.no_grad():
            gen_batch_size = script_args.gen_bsize if hasattr(script_args, 'gen_bsize') else 32
            all_responses = grpo_trainer.generate(
                expanded_queries,
                batch_size=gen_batch_size,
                return_prompt=False,
                **generation_kwargs
            )
        
        timing["time/grpo/generation"] = time.time() - t
        
        # Decode responses
        response_texts = tokenizer.batch_decode(all_responses, skip_special_tokens=True)
        query_texts = tokenizer.batch_decode(all_queries, skip_special_tokens=True)
        
        # Create full texts for reward model (query + response)
        full_texts = [q + r for q, r in zip(query_texts, response_texts)]
        
        if epoch == 1:
            print("\n=== GRPO Response Samples ===")
            for i in range(min(4, len(response_texts))):
                print(f"Response {i}: {response_texts[i][:100]}...")
            print("=" * 30)
        
        # Get rewards
        t = time.time()
        scores = get_reward_scores(reward_model, reward_tokenizer, full_texts, device)
        timing["time/grpo/reward"] = time.time() - t
        
        if epoch == 1:
            print(f"\n=== Reward Scores ===")
            print(f"Sample scores: {scores[:8]}")
            print(f"Mean: {mean(scores):.4f}, Std: {stdev(scores) if len(scores) > 1 else 0:.4f}")
            print("=" * 30)
        
        # Run GRPO step
        t = time.time()
        stats = grpo_trainer.step(
            queries=all_queries,
            responses=all_responses,
            scores=scores,
            gen_data_dir=script_args.gen_data_dir,
        )
        timing["time/grpo/step"] = time.time() - t
        
        # Log stats
        stats.update(timing)
        stats["time/grpo/total"] = time.time() - t0
        
        batch_log = {
            "query": query_texts,
            "response": response_texts,
        }
        
        grpo_trainer.log_stats(stats, batch_log, scores)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"\n[Step {epoch}] Reward mean: {mean(scores):.4f}, KL coef: {grpo_trainer.kl_ctl.value:.4f}")
        
        # Save checkpoint
        if script_args.save_freq and epoch % script_args.save_freq == 0:
            save_path = os.path.join(script_args.output_dir, f"step_{epoch}")
            grpo_trainer.save_pretrained(save_path)
    
    # Save final model
    final_path = os.path.join(script_args.output_dir, "final")
    grpo_trainer.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


def grpo_train_loop_with_validation(
    script_args,
    grpo_trainer,
    reward_model,
    tokenizer,
    reward_tokenizer,
    val_question_tensors,
    val_questions,
    min_length=20,
):
    """
    GRPO training loop with TracIn influence-based sample selection.
    
    This loop:
    1. Generates training responses
    2. Generates validation responses (for TracIn gradient computation)
    3. Computes per-sample influence scores via gradient inner products
    4. Trains only on samples with positive influence
    """
    print("=" * 50)
    print("Starting GRPO Training Loop with FULL TracIn")
    print(f"Steps: {script_args.steps}")
    print(f"Batch size: {script_args.batch_size}")
    print(f"Num generations per prompt: {script_args.num_generations}")
    print(f"Validation size: {len(val_question_tensors)}")
    print(f"TracIn batch size: {script_args.tracin_batch_size}")
    print(f"Val loss type: {script_args.val_loss_type}")
    print("=" * 50)
    
    # Generation kwargs
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": script_args.max_length - min_length,
        "temperature": script_args.temperature,
    }
    
    dataloader = grpo_trainer.dataloader
    device = grpo_trainer.current_device
    
    # Pre-generate validation responses once (or regenerate periodically)
    val_sample_size = min(script_args.tracin_val_batch_size, len(val_question_tensors))
    val_regenerate_freq = 10  # Regenerate validation responses every N steps
    
    # Initial validation generation
    print(f"\nGenerating validation responses for {val_sample_size} samples...")
    val_sample_indices = torch.randperm(len(val_question_tensors))[:val_sample_size]
    val_queries_batch = [val_question_tensors[idx].to(device) for idx in val_sample_indices]
    
    with torch.no_grad():
        val_responses_batch = grpo_trainer.generate(
            val_queries_batch,
            batch_size=val_sample_size,
            return_prompt=False,
            **generation_kwargs
        )
    
    # Compute validation rewards and advantages
    val_response_texts = tokenizer.batch_decode(val_responses_batch, skip_special_tokens=True)
    val_query_texts = [val_questions[idx] for idx in val_sample_indices]
    val_full_texts = [q + r for q, r in zip(val_query_texts, val_response_texts)]
    val_scores = get_reward_scores(reward_model, reward_tokenizer, val_full_texts, device)
    
    # Compute validation model inputs for gradient computation
    val_model_inputs = grpo_trainer.prepare_model_inputs(val_queries_batch, val_responses_batch)
    val_logprobs, val_masks = grpo_trainer.compute_logprobs(
        grpo_trainer.model, val_queries_batch, val_responses_batch, val_model_inputs
    )
    val_scores_tensor = torch.tensor(val_scores, device=device, dtype=torch.float32)
    
    # FREE temporary tensors used during initial validation setup
    del val_response_texts, val_query_texts, val_full_texts, val_scores
    del val_model_inputs, val_sample_indices
    gc.collect()
    torch.cuda.empty_cache()
    
    # Handle NaN in validation scores
    if torch.isnan(val_scores_tensor).any():
        print("WARNING: Validation scores contain NaN! Replacing with 0.")
        val_scores_tensor = torch.nan_to_num(val_scores_tensor, nan=0.0)
    
    val_advantages = grpo_trainer.compute_group_advantages(val_scores_tensor, 1)  # No grouping for validation
    
    # Handle NaN in validation advantages
    if torch.isnan(val_advantages).any():
        print("WARNING: Validation advantages contain NaN! Replacing with 0.")
        val_advantages = torch.nan_to_num(val_advantages, nan=0.0)
    
    print(f"Validation scores: mean={val_scores_tensor.mean():.4f}, std={val_scores_tensor.std():.4f}")
    print(f"Validation advantages: mean={val_advantages.mean():.4f}, std={val_advantages.std():.4f}")
    
    epoch = 0
    all_ghost_ips = []  # Track influence scores across training
    
    for batch in tqdm(dataloader, desc="GRPO TracIn Training"):
        if epoch >= script_args.steps:
            break
        epoch += 1
        
        timing = {}
        t0 = time.time()
        
        # Periodically regenerate validation responses for diversity
        if epoch % val_regenerate_freq == 0:
            print(f"\n[Step {epoch}] Regenerating validation responses...")
            
            # FREE old validation tensors BEFORE creating new ones
            try:
                del val_responses_batch, val_logprobs, val_masks, val_advantages
                del val_queries_batch, val_scores_tensor
            except NameError:
                pass  # First iteration or already deleted
            gc.collect()
            torch.cuda.empty_cache()
            
            val_sample_indices = torch.randperm(len(val_question_tensors))[:val_sample_size]
            val_queries_batch = [val_question_tensors[idx].to(device) for idx in val_sample_indices]
            
            with torch.no_grad():
                val_responses_batch = grpo_trainer.generate(
                    val_queries_batch,
                    batch_size=val_sample_size,
                    return_prompt=False,
                    **generation_kwargs
                )
            
            val_response_texts = tokenizer.batch_decode(val_responses_batch, skip_special_tokens=True)
            val_query_texts = [val_questions[idx] for idx in val_sample_indices]
            val_full_texts = [q + r for q, r in zip(val_query_texts, val_response_texts)]
            val_scores = get_reward_scores(reward_model, reward_tokenizer, val_full_texts, device)
            
            val_model_inputs = grpo_trainer.prepare_model_inputs(val_queries_batch, val_responses_batch)
            val_logprobs, val_masks = grpo_trainer.compute_logprobs(
                grpo_trainer.model, val_queries_batch, val_responses_batch, val_model_inputs
            )
            val_scores_tensor = torch.tensor(val_scores, device=device, dtype=torch.float32)
            
            # FREE temporary tensors used during regeneration
            del val_response_texts, val_query_texts, val_full_texts, val_scores
            del val_model_inputs, val_sample_indices
            gc.collect()
            torch.cuda.empty_cache()
            
            # Handle NaN in validation scores
            if torch.isnan(val_scores_tensor).any():
                print("WARNING: Validation scores contain NaN! Replacing with 0.")
                val_scores_tensor = torch.nan_to_num(val_scores_tensor, nan=0.0)
            
            val_advantages = grpo_trainer.compute_group_advantages(val_scores_tensor, 1)
            
            # Handle NaN in validation advantages
            if torch.isnan(val_advantages).any():
                print("WARNING: Validation advantages contain NaN! Replacing with 0.")
                val_advantages = torch.nan_to_num(val_advantages, nan=0.0)
            
            print(f"New validation scores: mean={val_scores_tensor.mean():.4f}")
        
        # Get query tensors
        question_tensors = batch["input_ids"]
        queries_text = batch.get("query", tokenizer.batch_decode(question_tensors, skip_special_tokens=True))
        
        # Generate multiple responses per query using BATCHED generation
        t = time.time()
        all_queries = []
        
        # Expand queries: repeat each query num_generations times
        expanded_queries = []
        for q_tensor in question_tensors:
            for _ in range(script_args.num_generations):
                expanded_queries.append(q_tensor.to(device))
                all_queries.append(q_tensor)
        
        # Batched generation - process all at once (much faster!)
        with torch.no_grad():
            gen_batch_size = script_args.gen_bsize if hasattr(script_args, 'gen_bsize') else 32
            all_responses = grpo_trainer.generate(
                expanded_queries,
                batch_size=gen_batch_size,
                return_prompt=False,
                **generation_kwargs
            )
        
        timing["time/grpo/generation"] = time.time() - t
        
        # Decode responses
        response_texts = tokenizer.batch_decode(all_responses, skip_special_tokens=True)
        query_texts = tokenizer.batch_decode(all_queries, skip_special_tokens=True)
        
        # Create full texts for reward model
        full_texts = [q + r for q, r in zip(query_texts, response_texts)]
        
        # Get rewards
        t = time.time()
        scores = get_reward_scores(reward_model, reward_tokenizer, full_texts, device)
        timing["time/grpo/reward"] = time.time() - t
        
        # Run GRPO TracIn step
        t = time.time()
        stats, ghost_ip = grpo_trainer.step_tracin(
            queries=all_queries,
            responses=all_responses,
            scores=scores,
            val_queries=val_queries_batch,
            val_responses=val_responses_batch,
            val_advantages=val_advantages,
            val_logprobs=val_logprobs,
            val_masks=val_masks,
            gen_data_dir=script_args.gen_data_dir,
        )
        timing["time/grpo/step"] = time.time() - t
        all_ghost_ips.extend(ghost_ip)
        
        # Log stats
        stats.update(timing)
        stats["time/grpo/total"] = time.time() - t0
        
        batch_log = {
            "query": query_texts,
            "response": response_texts,
        }
        
        grpo_trainer.log_stats(stats, batch_log, scores)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"\n[Step {epoch}] Train reward: {mean(scores):.4f}")
            print(f"  TracIn selection: {stats.get('tracin/num_selected', 0)} / {len(all_queries)}")
            print(f"  Mean IP: {stats.get('tracin/mean_ip', 0):.6f}")
            print(f"  KL coef: {grpo_trainer.kl_ctl.value:.4f}")
        
        # =====================
        # MEMORY CLEANUP (per step) - Critical for preventing OOM!
        # =====================
        # Delete GPU tensors from this iteration
        try:
            del all_queries, expanded_queries, all_responses
        except NameError:
            pass
        try:
            del question_tensors, queries_text
        except NameError:
            pass
        try:
            del full_texts, query_texts, response_texts
        except NameError:
            pass
        try:
            del stats, batch_log, timing
        except NameError:
            pass
        try:
            del scores
        except NameError:
            pass
        
        # Note: validation variables (val_responses_batch, val_logprobs, etc.) 
        # are intentionally kept for reuse across steps
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
        
        # Save checkpoint
        if script_args.save_freq and epoch % script_args.save_freq == 0:
            save_path = os.path.join(script_args.output_dir, f"step_{epoch}")
            grpo_trainer.save_pretrained(save_path)
    
    # Save final model
    final_path = os.path.join(script_args.output_dir, "final")
    grpo_trainer.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save TracIn statistics
    import json
    tracin_stats = {
        "total_samples": len(all_ghost_ips),
        "positive_ratio": sum(1 for ip in all_ghost_ips if ip > 0) / len(all_ghost_ips) if all_ghost_ips else 0,
        "mean_ip": mean(all_ghost_ips) if all_ghost_ips else 0,
    }
    with open(os.path.join(script_args.output_dir, "tracin_stats.json"), "w") as f:
        json.dump(tracin_stats, f, indent=2)
    print(f"TracIn stats saved: {tracin_stats}")


# Main execution
if __name__ == "__main__":
    parser = HfArgumentParser(GRPOScriptArguments)
    script_args: GRPOScriptArguments = parser.parse_args_into_dataclasses()[0]
    
    # Initialize wandb
    wandb.init(
        project=script_args.wandb_project,
        name=script_args.run_name,
        config=vars(script_args)
    )
    
    set_seed(script_args.seed)
    
    # Ensure output directory ends with /
    if script_args.output_dir[-1] != "/":
        script_args.output_dir = script_args.output_dir + "/"
    
    print("Loading GRPO models...")
    
    # Load models
    if "function" in script_args.reward_model_name:
        config, tokenizer, model, optimizer = load_models_grpo(script_args, "grpo")
        reward_model = None
        reward_tokenizer = None
    else:
        config, tokenizer, model, optimizer, reward_model, reward_tokenizer = load_models_grpo(script_args)
    
    print("Models loaded successfully!")
    print(f"Model: {script_args.model_name}")
    print(f"Reward model: {script_args.reward_model_name}")
    
    # Load dataset
    print(f"\nLoading dataset: {script_args.dataset_name}")
    rmformat = anscat
    
    if "toxicity" in script_args.dataset_name:
        dataset, valid_dataset = build_toxicity_promptdata(
            tokenizer,
            num_samples=script_args.val_size,
            seed=script_args.seed,
            val_strategy=script_args.val_strategy
        )
        val_question_tensors = valid_dataset['input_ids']
        val_questions = valid_dataset['query']
    else:
        raise ValueError(f"Dataset {script_args.dataset_name} not supported for GRPO training. Use 'toxicity'.")
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Validation size: {len(val_question_tensors)}")
    print(f"Sample data: {dataset[0]}")
    
    # Initialize GRPO Trainer
    grpo_trainer = GRPOTrainer(
        config=config,
        model=model,
        ref_model=None,  # Will be created internally or use PEFT disable_adapter
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )
    
    # Print trainable parameters
    trainable_params = [
        n for n, p in grpo_trainer.model.named_parameters()
        if p.requires_grad
    ]
    print(f"\n--- Trainable Parameters: {len(trainable_params)} ---")
    
    # Run training loop
    if script_args.tracin:
        if script_args.with_validation:
            print("\nRunning GRPO with TracIn validation...")
            grpo_train_loop_with_validation(
                script_args,
                grpo_trainer,
                reward_model,
                tokenizer,
                reward_tokenizer,
                val_question_tensors,
                val_questions,
                min_length=script_args.min_length,
            )
        else:
            print("\nRunning GRPO with TracIn (valid=train)...")
            grpo_train_loop(
                script_args,
                grpo_trainer,
                reward_model,
                tokenizer,
                reward_tokenizer,
                min_length=script_args.min_length,
            )
    else:
        print("\nRunning standard GRPO training...")
        grpo_train_loop(
            script_args,
            grpo_trainer,
            reward_model,
            tokenizer,
            reward_tokenizer,
            min_length=script_args.min_length,
        )
    
    wandb.finish()
