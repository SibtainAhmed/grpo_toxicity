# GRPO Training Script for Toxicity Reduction
# Based on train_rlhf.py but adapted for Group Relative Policy Optimization

import os
import sys
import time
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
import wandb
from statistics import mean, stdev

# Add scripts directory to path for local imports
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grpo_trainer import GRPOTrainer
from grpo_config import GRPOConfig

from rlhfutils.rl_utils import (
    GRPOScriptArguments,
    load_models_grpo,
)

from rlhfutils.data import (
    build_toxicity_promptdata,
    collator,
    anscat,
)

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
        # For RoBERTa hate speech model, class 0 is "nothate", class 1 is "hate"
        # We want to reward "nothate" (less toxic), so use negative of hate probability
        probs = torch.softmax(outputs.logits, dim=-1)
        # Higher score = less toxic
        scores = probs[:, 0] - probs[:, 1]  # nothate - hate
    
    return scores.cpu().tolist()


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
        
        # Generate multiple responses per query
        t = time.time()
        all_responses = []
        all_queries = []
        
        with torch.no_grad():
            for q_tensor in question_tensors:
                for _ in range(script_args.num_generations):
                    response = grpo_trainer.generate(
                        q_tensor.to(device),
                        return_prompt=False,
                        **generation_kwargs
                    )
                    if isinstance(response, torch.Tensor):
                        all_responses.append(response.squeeze(0))
                    else:
                        all_responses.append(response[0])
                    all_queries.append(q_tensor)
        
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
    GRPO training loop with TracIn-style validation.
    Uses validation set to compute influence scores for sample selection.
    """
    print("=" * 50)
    print("Starting GRPO Training Loop with TracIn Validation")
    print(f"Steps: {script_args.steps}")
    print(f"Batch size: {script_args.batch_size}")
    print(f"Num generations per prompt: {script_args.num_generations}")
    print(f"Validation size: {len(val_question_tensors)}")
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
    for batch in tqdm(dataloader, desc="GRPO Training with TracIn"):
        if epoch >= script_args.steps:
            break
        epoch += 1
        
        timing = {}
        t0 = time.time()
        
        # Get query tensors
        question_tensors = batch["input_ids"]
        queries_text = batch.get("query", tokenizer.batch_decode(question_tensors, skip_special_tokens=True))
        
        # Generate multiple responses per query
        t = time.time()
        all_responses = []
        all_queries = []
        
        with torch.no_grad():
            for q_tensor in question_tensors:
                for _ in range(script_args.num_generations):
                    response = grpo_trainer.generate(
                        q_tensor.to(device),
                        return_prompt=False,
                        **generation_kwargs
                    )
                    if isinstance(response, torch.Tensor):
                        all_responses.append(response.squeeze(0))
                    else:
                        all_responses.append(response[0])
                    all_queries.append(q_tensor)
        
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
        
        # For TracIn: also evaluate on validation set periodically
        if epoch % 10 == 0:
            t = time.time()
            # Generate responses for validation set (sample)
            val_sample_size = min(16, len(val_question_tensors))
            val_sample_indices = torch.randperm(len(val_question_tensors))[:val_sample_size]
            
            val_responses = []
            with torch.no_grad():
                for idx in val_sample_indices:
                    q_tensor = val_question_tensors[idx]
                    response = grpo_trainer.generate(
                        q_tensor.to(device),
                        return_prompt=False,
                        **generation_kwargs
                    )
                    if isinstance(response, torch.Tensor):
                        val_responses.append(response.squeeze(0))
                    else:
                        val_responses.append(response[0])
            
            val_response_texts = tokenizer.batch_decode(val_responses, skip_special_tokens=True)
            val_query_texts = [val_questions[idx] for idx in val_sample_indices]
            val_full_texts = [q + r for q, r in zip(val_query_texts, val_response_texts)]
            
            val_scores = get_reward_scores(reward_model, reward_tokenizer, val_full_texts, device)
            timing["time/grpo/validation"] = time.time() - t
            
            print(f"\n[Step {epoch}] Validation reward: {mean(val_scores):.4f}")
        
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
            print(f"[Step {epoch}] Train reward: {mean(scores):.4f}, KL coef: {grpo_trainer.kl_ctl.value:.4f}")
        
        # Save checkpoint
        if script_args.save_freq and epoch % script_args.save_freq == 0:
            save_path = os.path.join(script_args.output_dir, f"step_{epoch}")
            grpo_trainer.save_pretrained(save_path)
    
    # Save final model
    final_path = os.path.join(script_args.output_dir, "final")
    grpo_trainer.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


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
