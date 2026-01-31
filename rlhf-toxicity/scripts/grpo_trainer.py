# Copyright 2022 The HuggingFace Team. All rights reserved.
# Adapted for GRPO (Group Relative Policy Optimization)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GRPO Trainer - Group Relative Policy Optimization

Key differences from PPO:
1. No value head - uses regular AutoModelForCausalLM
2. Generates multiple responses per prompt
3. Computes advantages as group-normalized rewards
4. Simpler loss function (no value loss)
"""
import copy
import inspect
import math
import os
import time
import typing
import warnings
from typing import Callable, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import Dataset
from packaging import version
from torch.optim import Adam
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)

from grpo_config import GRPOConfig


def set_seed(seed: int):
    """Set seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""
    def recurse(nest, prefix):
        res = {}
        if isinstance(nest, dict):
            for key, value in nest.items():
                new_key = f"{prefix}{sep}{key}" if prefix else key
                res.update(recurse(value, new_key))
        else:
            res[prefix] = nest
        return res
    return recurse(nested, "")


def masked_mean(tensor, mask, axis=None):
    """Compute mean of tensor with a mask."""
    if axis is not None:
        return (tensor * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (tensor * mask).sum() / mask.sum()


def masked_var(tensor, mask, unbiased=True):
    """Compute variance of tensor with a mask."""
    mean = masked_mean(tensor, mask)
    centered = tensor - mean
    variance = masked_mean(centered ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            return variance
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(tensor, mask, shift_mean=True):
    """Whiten (normalize) tensor with a mask."""
    mean = masked_mean(tensor, mask)
    var = masked_var(tensor, mask)
    whitened = (tensor - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def logprobs_from_logits(logits, labels, gather=True):
    """Compute per-token log probabilities from logits and labels."""
    logp = F.log_softmax(logits, dim=2)
    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def entropy_from_logits(logits):
    """Compute entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def clip_by_value(x, tensor_min, tensor_max):
    """Clip tensor by min and max values."""
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def convert_to_scalar(stats):
    """Convert tensors to scalars in a stats dictionary."""
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            stats[key] = value.item()
    return stats


class AdaptiveKLController:
    """Adaptive KL controller as described in Ziegler et al."""

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class GRPOTrainer:
    """
    GRPO Trainer - Group Relative Policy Optimization
    
    Unlike PPO, GRPO:
    - Does not use a value head
    - Generates multiple responses per prompt
    - Uses group-normalized rewards as advantages
    """

    def __init__(
        self,
        config: GRPOConfig = None,
        model = None,
        ref_model = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize GRPOTrainer.

        Args:
            config: Configuration object for GRPOTrainer
            model: Hugging Face transformer model (no value head needed)
            ref_model: Reference model for KL penalty
            tokenizer: Tokenizer for encoding/decoding
            dataset: Training dataset
            optimizer: Optimizer (default: Adam)
            data_collator: Data collator function
            lr_scheduler: Learning rate scheduler
        """
        self.config = config
        set_seed(config.seed)

        # Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"

        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_grpo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )

        self.model = model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)

        # Reference model for KL computation
        if ref_model is not None:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None  # PEFT models can disable adapter for reference
        else:
            # Create reference model by copying
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        self.tokenizer = tokenizer
        self.dataset = dataset
        self._signature_columns = None

        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        else:
            self.dataloader = None

        self.config.backward_batch_size = self.config.mini_batch_size * self.config.gradient_accumulation_steps

        # Initialize optimizer
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        # KL controller
        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )

        if self.ref_model is not None and not self.is_peft_model:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        self.is_distributed = self.accelerator.distributed_type != "NO"
        self.current_device = self.accelerator.device

        # For saving
        self.save_cnt = 0

    def prepare_dataloader(self, dataset, data_collator=None):
        """Prepare the dataloader for training."""
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size // self.config.num_generations,  # Adjust for multiple generations
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def _remove_unused_columns(self, dataset):
        """Remove columns not used by the model."""
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            columns = [k for k in signature_columns if k in dataset.column_names]
            dataset = dataset.remove_columns(ignored_columns)
        return dataset

    def _set_signature_columns_if_needed(self):
        """Set signature columns for dataset processing."""
        if self._signature_columns is None:
            self._signature_columns = ["input_ids", "attention_mask", "label", "query", "response"]

    @property
    def device(self):
        return self.current_device

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """Generate responses for query tensors."""
        if isinstance(query_tensor, List):
            return self._generate_batched(
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
        else:
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()
            response = self.accelerator.unwrap_model(self.model).generate(
                input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
            )
            if not return_prompt and not self.is_encoder_decoder:
                return response[:, query_tensor.shape[0]:]
            return response

    def _generate_batched(
        self,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        """Generate responses in batches."""
        outputs = []
        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            end_index = min(len(query_tensors), i + batch_size)
            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            generations = self.accelerator.unwrap_model(self.model).generate(
                **padded_inputs, **generation_kwargs
            )

            for generation, mask, query in zip(generations, padded_inputs["attention_mask"], batch):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum():]
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[len(query):]

                if remove_padding and self.tokenizer.eos_token_id is not None:
                    if output[-1] == self.tokenizer.eos_token_id:
                        output = output[:-1]
                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def generate_multiple(
        self,
        query_tensors: List[torch.Tensor],
        num_generations: int,
        **generation_kwargs,
    ):
        """
        Generate multiple responses per query for group-relative advantage computation.
        
        Args:
            query_tensors: List of query tensors
            num_generations: Number of responses to generate per query
            generation_kwargs: Generation parameters
            
        Returns:
            responses: List of response tensors (length = len(query_tensors) * num_generations)
            expanded_queries: Queries repeated to match responses
        """
        all_responses = []
        expanded_queries = []
        
        for query in query_tensors:
            # Generate multiple responses for this query
            for _ in range(num_generations):
                response = self.generate(query, return_prompt=False, **generation_kwargs)
                if isinstance(response, torch.Tensor):
                    all_responses.append(response.squeeze(0))
                else:
                    all_responses.append(response[0])
                expanded_queries.append(query)
        
        return all_responses, expanded_queries

    def compute_group_advantages(self, rewards: torch.Tensor, num_generations: int, normalize_batch: bool = True):
        """
        Compute group-relative advantages.
        
        For each prompt, the advantage of each response is computed as:
        advantage = (reward - mean(group_rewards)) / std(group_rewards)
        
        Args:
            rewards: Tensor of shape [batch_size * num_generations]
            num_generations: Number of generations per prompt
            normalize_batch: Whether to normalize advantages across entire batch (reduces variance)
            
        Returns:
            advantages: Tensor of shape [batch_size * num_generations]
        """
        # Reshape to [num_prompts, num_generations]
        rewards_grouped = rewards.view(-1, num_generations)
        
        # Compute mean and std per group
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True)
        
        # Use larger epsilon and clamp std to avoid division issues with small groups
        std = torch.clamp(std, min=0.1)  # Minimum std of 0.1 for stability
        
        # Normalize within each group
        advantages = (rewards_grouped - mean) / std
        
        # Flatten back
        advantages = advantages.view(-1)
        
        # IMPORTANT: Normalize advantages across entire batch (like PPO does)
        # This significantly reduces variance and stabilizes training
        if normalize_batch:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages

    def prepare_model_inputs(self, queries, responses):
        """Prepare model inputs from queries and responses."""
        input_ids = []
        attention_mask = []
        
        for query, response in zip(queries, responses):
            # Concatenate query and response
            full_seq = torch.cat([query, response])
            input_ids.append(full_seq)
            attention_mask.append(torch.ones_like(full_seq))
        
        # Pad sequences
        max_len = max(len(seq) for seq in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        
        for seq, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(seq)
            if self.tokenizer.padding_side == "left":
                padded_seq = F.pad(seq, (pad_len, 0), value=self.tokenizer.pad_token_id)
                padded_mask = F.pad(mask, (pad_len, 0), value=0)
            else:
                padded_seq = F.pad(seq, (0, pad_len), value=self.tokenizer.pad_token_id)
                padded_mask = F.pad(mask, (0, pad_len), value=0)
            padded_input_ids.append(padded_seq)
            padded_attention_mask.append(padded_mask)
        
        return {
            "input_ids": torch.stack(padded_input_ids).to(self.current_device),
            "attention_mask": torch.stack(padded_attention_mask).to(self.current_device),
        }

    def compute_logprobs(self, model, queries, responses, model_inputs):
        """Compute log probabilities for responses given queries."""
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        
        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get log probs for response tokens only
        all_logprobs = []
        all_masks = []
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            query_len = len(query)
            response_len = len(response)
            
            # Get logits for this sequence
            seq_logits = logits[i]
            
            # Compute log probs
            log_probs = F.log_softmax(seq_logits, dim=-1)
            
            # Get response token log probs (shift by 1 for autoregressive)
            response_start = query_len
            response_end = query_len + response_len
            
            # Get log probs of actual tokens
            response_logprobs = torch.gather(
                log_probs[response_start-1:response_end-1],
                dim=-1,
                index=input_ids[i, response_start:response_end].unsqueeze(-1)
            ).squeeze(-1)
            
            all_logprobs.append(response_logprobs)
            all_masks.append(torch.ones_like(response_logprobs))
        
        # Pad to same length
        max_len = max(lp.shape[0] for lp in all_logprobs)
        padded_logprobs = []
        padded_masks = []
        
        for lp, mask in zip(all_logprobs, all_masks):
            pad_len = max_len - lp.shape[0]
            padded_logprobs.append(F.pad(lp, (0, pad_len), value=0))
            padded_masks.append(F.pad(mask, (0, pad_len), value=0))
        
        return torch.stack(padded_logprobs), torch.stack(padded_masks)

    def _extract_response_logits(self, logits, queries, responses):
        """
        Extract only the response portion of logits for entropy computation.
        
        Args:
            logits: Full sequence logits [batch, full_seq_len, vocab_size]
            queries: List of query tensors
            responses: List of response tensors
            
        Returns:
            response_logits: Padded response logits [batch, max_response_len, vocab_size]
        """
        all_response_logits = []
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            query_len = len(query)
            response_len = len(response)
            
            # Extract response portion of logits (aligned with response tokens)
            # We use query_len-1 to query_len+response_len-1 because of autoregressive shift
            response_start = query_len - 1
            response_end = query_len + response_len - 1
            
            # Handle edge case where response_start might be negative
            if response_start < 0:
                response_start = 0
            if response_end > logits.shape[1]:
                response_end = logits.shape[1]
            
            response_logits = logits[i, response_start:response_end]
            all_response_logits.append(response_logits)
        
        # Pad to same length
        max_len = max(rl.shape[0] for rl in all_response_logits)
        vocab_size = logits.shape[-1]
        
        padded_response_logits = []
        for rl in all_response_logits:
            pad_len = max_len - rl.shape[0]
            if pad_len > 0:
                padding = torch.zeros(pad_len, vocab_size, device=rl.device, dtype=rl.dtype)
                rl = torch.cat([rl, padding], dim=0)
            padded_response_logits.append(rl)
        
        return torch.stack(padded_response_logits)

    def _extract_response_logits_consistent(self, logits, queries, responses, target_len):
        """
        Extract response logits with consistent padding to target_len.
        """
        all_response_logits = []
        vocab_size = logits.shape[-1]
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            query_len = len(query)
            response_len = len(response)
            
            response_start = query_len - 1
            response_end = query_len + response_len - 1
            
            if response_start < 0:
                response_start = 0
            if response_end > logits.shape[1]:
                response_end = logits.shape[1]
            
            response_logits = logits[i, response_start:response_end]
            
            # Pad to target_len
            pad_len = target_len - response_logits.shape[0]
            if pad_len > 0:
                padding = torch.zeros(pad_len, vocab_size, device=response_logits.device, dtype=response_logits.dtype)
                response_logits = torch.cat([response_logits, padding], dim=0)
            elif pad_len < 0:
                # Truncate if longer
                response_logits = response_logits[:target_len]
            
            all_response_logits.append(response_logits)
        
        return torch.stack(all_response_logits)

    def _compute_logprobs_consistent(self, logits, queries, responses, model_inputs, target_len):
        """
        Compute log probabilities with consistent padding to target_len.
        """
        input_ids = model_inputs["input_ids"]
        all_logprobs = []
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            query_len = len(query)
            response_len = len(response)
            
            # Get logits for this sequence
            seq_logits = logits[i]
            
            # Compute log probs
            log_probs = F.log_softmax(seq_logits, dim=-1)
            
            # Get response token log probs (shift by 1 for autoregressive)
            response_start = query_len
            response_end = query_len + response_len
            
            # Ensure we don't go out of bounds
            if response_end > input_ids.shape[1]:
                response_end = input_ids.shape[1]
            if response_start >= response_end:
                # Edge case: empty response
                response_logprobs = torch.zeros(target_len, device=logits.device, dtype=logits.dtype)
            else:
                # Get log probs of actual tokens
                response_logprobs = torch.gather(
                    log_probs[response_start-1:response_end-1],
                    dim=-1,
                    index=input_ids[i, response_start:response_end].unsqueeze(-1)
                ).squeeze(-1)
                
                # Pad to target_len
                pad_len = target_len - response_logprobs.shape[0]
                if pad_len > 0:
                    response_logprobs = F.pad(response_logprobs, (0, pad_len), value=0)
                elif pad_len < 0:
                    response_logprobs = response_logprobs[:target_len]
            
            all_logprobs.append(response_logprobs)
        
        return torch.stack(all_logprobs)

    def compute_rewards_with_kl(self, scores, logprobs, ref_logprobs, masks):
        """
        Compute rewards with KL penalty.
        
        reward = score - kl_coef * kl
        where kl = logprobs - ref_logprobs
        """
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl
        
        # Apply score only to last token
        rewards = non_score_reward.clone()
        for i, score in enumerate(scores):
            # Find last non-padded position
            last_pos = masks[i].sum().long() - 1
            rewards[i, last_pos] += score
        
        return rewards, non_score_reward

    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        gen_data_dir: str = None,
    ):
        """
        Run a GRPO optimization step.
        
        Args:
            queries: List of query tensors
            responses: List of response tensors
            scores: List of reward scores
            gen_data_dir: Directory to save generated data
            
        Returns:
            stats: Dictionary of training statistics
        """
        timing = {}
        t0 = time.time()
        
        bs = len(queries)
        
        # Prepare model inputs
        t = time.time()
        model_inputs = self.prepare_model_inputs(queries, responses)
        timing["time/grpo/prepare_inputs"] = time.time() - t
        
        # Compute log probabilities
        t = time.time()
        self.model.eval()
        with torch.no_grad():
            logprobs, masks = self.compute_logprobs(self.model, queries, responses, model_inputs)
            
            # Get reference log probs
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model), "disable_adapter"
            ):
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_logprobs, _ = self.compute_logprobs(self.model, queries, responses, model_inputs)
            elif self.ref_model is not None:
                ref_logprobs, _ = self.compute_logprobs(self.ref_model, queries, responses, model_inputs)
            else:
                ref_logprobs = logprobs.clone()
        
        timing["time/grpo/forward_pass"] = time.time() - t
        
        # Convert scores to tensor
        scores_tensor = torch.tensor(scores, device=self.current_device, dtype=torch.float32)
        
        # Compute rewards with KL penalty (for loss computation)
        t = time.time()
        rewards, non_score_reward = self.compute_rewards_with_kl(scores_tensor, logprobs, ref_logprobs, masks)
        timing["time/grpo/compute_rewards"] = time.time() - t
        
        # Compute group-relative advantages
        t = time.time()
        # IMPORTANT: Use RAW SCORES for advantage computation, not KL-penalized rewards!
        # This is the key difference - GRPO computes relative advantages from original reward signals
        # KL penalty is already applied in the loss through the ratio clipping
        advantages = self.compute_group_advantages(scores_tensor, self.config.num_generations)
        # Expand advantages to sequence level
        advantages_expanded = advantages.unsqueeze(-1).expand_as(logprobs)
        timing["time/grpo/compute_advantages"] = time.time() - t
        
        # Debug logging (every 10 steps)
        if self.save_cnt % 10 == 0:
            print(f"\n[GRPO Debug Step {self.save_cnt}]")
            print(f"  Scores: mean={scores_tensor.mean():.4f}, std={scores_tensor.std():.4f}, min={scores_tensor.min():.4f}, max={scores_tensor.max():.4f}")
            print(f"  Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
            print(f"  KL coef: {self.kl_ctl.value:.4f}")
        
        # Save generated data if requested
        if gen_data_dir is not None:
            os.makedirs(gen_data_dir, exist_ok=True)
            torch.save({
                "queries": queries,
                "responses": responses,
                "scores": scores_tensor.cpu(),
                "rewards": rewards.cpu(),
                "advantages": advantages.cpu(),
                "logprobs": logprobs.cpu(),
                "ref_logprobs": ref_logprobs.cpu(),
                "masks": masks.cpu(),
                "kl_ctl_value": self.kl_ctl.value,
            }, f'{gen_data_dir}/grpo_samples_seed-{self.config.seed}_{self.save_cnt}.pt')
            print(f'File saved to {gen_data_dir}/grpo_samples_seed-{self.config.seed}_{self.save_cnt}.pt')
            self.save_cnt += 1
        
        # Training loop
        t = time.time()
        all_stats = []
        self.model.train()
        
        # Store response lengths for consistent indexing
        response_lengths = [len(r) for r in responses]
        max_response_len = logprobs.shape[1]  # Use cached logprobs padding length
        
        for epoch in range(self.config.grpo_epochs):
            # Shuffle indices for mini-batching
            indices = torch.randperm(bs)
            
            for start_idx in range(0, bs, self.config.mini_batch_size):
                end_idx = min(start_idx + self.config.mini_batch_size, bs)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data (use cached tensors with consistent padding)
                mb_logprobs_old = logprobs[batch_indices].detach()
                mb_advantages = advantages_expanded[batch_indices].detach()
                mb_masks = masks[batch_indices]
                mb_queries = [queries[i] for i in batch_indices]
                mb_responses = [responses[i] for i in batch_indices]
                
                # Use the ORIGINAL model_inputs with batch indexing for consistent sequence lengths
                mb_model_inputs = {
                    "input_ids": model_inputs["input_ids"][batch_indices],
                    "attention_mask": model_inputs["attention_mask"][batch_indices],
                }
                
                # Forward pass
                outputs = self.model(
                    input_ids=mb_model_inputs["input_ids"],
                    attention_mask=mb_model_inputs["attention_mask"],
                )
                logits = outputs.logits
                
                # Compute new log probs with consistent padding
                mb_logprobs_new = self._compute_logprobs_consistent(
                    logits, mb_queries, mb_responses, mb_model_inputs, max_response_len
                )
                
                # Extract response-only logits for entropy computation (with consistent padding)
                response_logits = self._extract_response_logits_consistent(
                    logits, mb_queries, mb_responses, max_response_len
                )
                
                # Compute loss
                pg_loss, stats = self.loss(
                    mb_logprobs_old,
                    response_logits,
                    mb_logprobs_new,
                    mb_masks,
                    mb_advantages,
                )
                
                # Backward pass
                self.accelerator.backward(pg_loss)
                
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                all_stats.append(stats)
        
        timing["time/grpo/optimization"] = time.time() - t
        
        # Aggregate stats
        stats = {}
        for key in all_stats[0].keys():
            values = [s[key] for s in all_stats if key in s]
            if len(values) > 0:
                if isinstance(values[0], torch.Tensor):
                    stats[key] = torch.stack(values).mean()
                else:
                    stats[key] = np.mean(values)
        
        # Update KL controller
        kl = ((logprobs - ref_logprobs) * masks).sum(dim=-1).mean()
        self.kl_ctl.update(kl.item(), bs)
        
        # Record step stats
        stats.update(self.record_step_stats(
            kl_coef=self.kl_ctl.value,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            masks=masks,
            scores=scores_tensor,
            rewards=rewards,
        ))
        stats.update(timing)
        stats["time/grpo/total"] = time.time() - t0
        
        return stats

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        entropy_coef: float = 0.01,
    ):
        """
        Calculate GRPO policy loss (no value loss).
        
        Args:
            old_logprobs: Log probabilities from old policy
            logits: Logits from current policy
            logprobs: Log probabilities from current policy
            mask: Attention mask
            advantages: Group-relative advantages
            entropy_coef: Coefficient for entropy bonus (encourages exploration)
        """
        # Policy gradient with clipping
        ratio = torch.exp(logprobs - old_logprobs)
        
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.config.cliprange,
            1.0 + self.config.cliprange
        )
        
        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)
        
        # Check for high ratio
        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"Average ratio ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
        
        # Entropy bonus - encourages exploration and reduces variance
        entropy = masked_mean(entropy_from_logits(logits), mask)
        
        # Total loss = policy loss - entropy bonus (we want to maximize entropy)
        total_loss = pg_loss - entropy_coef * entropy
        
        # Approximate KL
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)
        
        stats = {
            "loss/policy": pg_loss.detach(),
            "loss/entropy": (-entropy_coef * entropy).detach(),
            "loss/total": total_loss.detach(),
            "policy/entropy": entropy.detach(),
            "policy/approxkl": approxkl.detach(),
            "policy/policykl": policykl.detach(),
            "policy/clipfrac": pg_clipfrac.detach(),
            "policy/advantages_mean": masked_mean(advantages, mask).detach(),
            "policy/advantages_std": masked_var(advantages, mask).sqrt().detach(),
            "policy/ratio": avg_ratio,
        }
        
        return total_loss, stats

    def record_step_stats(self, kl_coef: float, **data):
        """Record training step statistics."""
        stats = {"objective/kl_coef": kl_coef}
        
        masks = data.get("masks")
        logprobs = data.get("logprobs")
        ref_logprobs = data.get("ref_logprobs")
        
        if ref_logprobs is not None and masks is not None:
            kl_list = ((logprobs - ref_logprobs) * masks).sum(dim=-1)
            mean_kl = kl_list.mean()
            stats["objective/kl"] = mean_kl
        
        if "scores" in data:
            stats["env/reward_mean"] = data["scores"].mean()
            stats["env/reward_std"] = data["scores"].std()
        
        if "rewards" in data and masks is not None:
            stats["ppo/returns/mean"] = masked_mean(data["rewards"], masks)
        
        return stats

    def log_stats(self, stats, batch, rewards):
        """Log training stats to tracker."""
        if self.accelerator.is_main_process:
            logs = {}
            
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards).to(self.current_device)
            
            if self.config.log_with == "wandb":
                import wandb
                if "query" in batch and "response" in batch:
                    table_rows = [
                        list(r) for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())
                    ]
                    logs["game_log"] = wandb.Table(
                        columns=["query", "response", "reward"],
                        rows=table_rows
                    )
            
            logs.update(stats)
            logs["env/reward_mean"] = rewards.mean().item()
            logs["env/reward_std"] = rewards.std().item()
            
            # Convert tensors to scalars
            for k, v in logs.items():
                if isinstance(v, torch.Tensor):
                    logs[k] = v.item() if v.numel() == 1 else v.mean().item()
            
            self.accelerator.log(logs)

    def save_pretrained(self, output_dir: str):
        """Save model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        self.accelerator.unwrap_model(self.model).save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
