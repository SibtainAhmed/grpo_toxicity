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

import contextlib
@contextlib.contextmanager
def ghost_mode(optimizer):
    """Context manager to run backward pass without updating weights."""
    _orig_step = optimizer.step
    optimizer.step = lambda *a, **k: None
    try:
        yield
    finally:
        optimizer.step = _orig_step


def log_gpu_memory(step_name: str, verbose: bool = False):
    """Log GPU memory usage for debugging memory leaks."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        if verbose:
            print(f"[Memory] {step_name}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        return allocated, reserved
    return 0, 0


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
        
        # ========================================
        # TracIn: Ghost gradient buffers and hooks
        # ========================================
        # Buffers for per-sample gradient computation (LoRA layers)
        self._xs   = {}  # input activations to lora_A
        self._hs   = {}  # output of lora_A (hidden states)
        self._gAs  = {}  # gradients of lora_A
        self._gBs  = {}  # gradients of lora_B
        
        # No value head in GRPO, so no _vxs, _vgs, _bgs needed
        
        self._record_ghost = False
        self._train_xs = {}
        self._train_hs = {}
        self._train_gAs = {}
        self._train_gBs = {}
        
        # Install hooks on LoRA layers for TracIn gradient tracking
        self._install_ghost_hooks()

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

    def _install_ghost_hooks(self):
        """Install forward/backward hooks on LoRA layers for TracIn gradient tracking."""
        try:
            from peft.tuners.lora import LoraLayer
        except ImportError:
            print("PEFT not installed or no LoRA layers - TracIn hooks not installed")
            return
        
        for name, module in self.model.named_modules():
            # Hook LoRA adapters for ghost gradient computation
            if isinstance(module, LoraLayer):
                if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                    continue
                    
                if not hasattr(module.lora_A, 'default') or not hasattr(module.lora_B, 'default'):
                    continue
                
                r = module.lora_A.default.weight.shape[0]  # LoRA rank
                
                # Initialize buffers for this layer
                self._xs[name]  = []
                self._hs[name]  = []
                self._gAs[name] = []
                self._gBs[name] = []
                
                d_in_loraA = module.lora_A.default.weight.shape[1]
                d_out_loraB = module.lora_B.default.weight.shape[0]
                
                # Forward hook on A: capture input x and output h = A @ x
                def fwd_A(mod, inp, out, nm=name, d_in=d_in_loraA, rank=r):
                    if not self._record_ghost:
                        return
                    x, = inp  # [B, S, d_in]
                    h = out   # [B, S, r]
                    self._xs[nm].append(x.detach().clone())
                    self._hs[nm].append(h.detach().clone())
                
                module.lora_A.default.register_forward_hook(fwd_A)
                
                # Backward hook on B: capture gradients g_A = B^T @ g_out, and g_B
                def bwd_B(mod, grad_inp, grad_out, nm=name, d_out=d_out_loraB, rank=r):
                    if not self._record_ghost:
                        return
                    g_h = grad_inp[0]  # [B, S, r] - gradient w.r.t. lora_A output
                    g_o = grad_out[0]  # [B, S, d_out] - gradient w.r.t. lora_B output
                    self._gAs[nm].append(g_h.detach().clone())
                    self._gBs[nm].append(g_o.detach().clone())
                
                module.lora_B.default.register_full_backward_hook(bwd_B)
        
        print(f"TracIn hooks installed on {len(self._xs)} LoRA layers")

    def _clear_ghost_buffers(self):
        """Clear ghost gradient buffers."""
        for buf in (self._xs, self._hs, self._gAs, self._gBs):
            for name in buf:
                buf[name] = []
    
    def _save_train_buffers(self):
        """Save training buffers before computing validation gradients."""
        self._train_xs = copy.deepcopy(self._xs)
        self._train_hs = copy.deepcopy(self._hs)
        self._train_gAs = copy.deepcopy(self._gAs)
        self._train_gBs = copy.deepcopy(self._gBs)
        
    def compute_ghost_grad_norm(self):
        """Compute per-sample gradient norms for TracIn."""
        sample_norms = np.zeros((self.config.batch_size,), dtype=np.float32)
        
        for name in self._xs:
            if len(self._xs[name]) == 0:
                continue
            # Concatenate all micro-batches
            X = torch.cat(self._xs[name], dim=0)    # [N, S, d_in]
            H = torch.cat(self._hs[name], dim=0)    # [N, S, r]
            GAt = torch.cat(self._gAs[name], dim=0) # [N, S, r]
            GBt = torch.cat(self._gBs[name], dim=0) # [N, S, d_out]
            
            # Flatten sequence dimension for norm computation
            N = X.shape[0]
            X = X.view(N, -1)    # [N, S*d_in]
            H = H.view(N, -1)    # [N, S*r]
            GAt = GAt.view(N, -1)
            GBt = GBt.view(N, -1)
            
            for i in range(min(N, len(sample_norms))):
                # Per-sample gradient norm: ||grad||^2 = ||gA @ x^T||^2 + ||gB @ h^T||^2
                block_A = ((GAt[i] @ GAt[i].T) * (X[i] @ X[i].T)).sum()
                block_B = ((GBt[i] @ GBt[i].T) * (H[i] @ H[i].T)).sum()
                sample_norms[i] += (block_A + block_B).item()
        
        return sample_norms.tolist()

    def compute_ghost_inner_product_matrix_op(self):
        """
        Compute TracIn influence scores using PPO step_part_I's approach:
        - Training gradients use TRAINING activations: train_gA.T @ train_X
        - Validation gradients use VALIDATION activations: val_gA.T @ val_X
        
        This matches PPO's compute_ghost_inner_product_diff_train_val_matrix_op exactly.
        """
        sample_IP = torch.zeros((self.config.batch_size,), device=self.accelerator.device)
        
        def compute_sample_ip_train_vec(GAt, GBt, train_GAt, train_GBt, X, H, train_X, train_H):
            """
            PPO step_part_I style with SEPARATE activations:
            - Validation: P_A[j] = GAt[j].T @ X[j] (validation activations)
            - Training: Q_A[i] = train_GAt[i].T @ train_X[i] (training activations)
            """
            # Convert to float32 for numerical stability
            GAt = GAt.float()
            GBt = GBt.float()
            train_GAt = train_GAt.float()
            train_GBt = train_GBt.float()
            X = X.float()
            H = H.float()
            train_X = train_X.float()
            train_H = train_H.float()
            
            # Block A:
            # P_A[j] = GAt[j].T @ X[j]  → shape [n_val, r, d_in]
            P_A = torch.matmul(GAt.transpose(1,2), X)
            S_A = P_A.sum(dim=0)       # [r, d_in]
            # Q_A[i] = train_GAt[i].T @ train_X[i]  → [n_train, r, d_in]
            Q_A = torch.matmul(train_GAt.transpose(1,2), train_X)
            sample_A = (Q_A * S_A).sum(dim=(1,2))  # → [n_train]

            # Block B:
            # P_B[j] = GBt[j].T @ H[j]  → [n_val, d_out, r]
            P_B = torch.matmul(GBt.transpose(1,2), H)
            S_B = P_B.sum(dim=0)       # [d_out, r]
            # Q_B[i] = train_GBt[i].T @ train_H[i]  → [n_train, d_out, r]
            Q_B = torch.matmul(train_GBt.transpose(1,2), train_H)
            sample_B = (Q_B * S_B).sum(dim=(1,2))  # → [n_train]

            return sample_A + sample_B
        
        # Loop over every LoRA adapter
        for name in self._train_xs:
            if len(self._train_xs.get(name, [])) == 0 or len(self._gAs.get(name, [])) == 0:
                continue
            
            # Training data (activations AND gradients)
            train_X = torch.cat(self._train_xs[name], dim=0)    # [N_train, S, d_in]
            train_H = torch.cat(self._train_hs[name], dim=0)    # [N_train, S, r]
            train_GAt = torch.cat(self._train_gAs[name], dim=0) # [N_train, S, r]
            train_GBt = torch.cat(self._train_gBs[name], dim=0) # [N_train, S, d_out]
            
            # Validation data (activations AND gradients from validation forward/backward)
            X   = torch.cat(self._xs[name], dim=0) if len(self._xs[name]) > 0 else None
            H   = torch.cat(self._hs[name], dim=0) if len(self._hs[name]) > 0 else None
            GAt = torch.cat(self._gAs[name], dim=0)
            GBt = torch.cat(self._gBs[name], dim=0)
            
            N_train = train_X.shape[0]
            
            if X is None or H is None:
                print(f"WARNING: No validation activations for layer {name}")
                continue
            
            sample_IP[:N_train] += compute_sample_ip_train_vec(
                GAt, GBt, train_GAt, train_GBt, X, H, train_X, train_H
            )
        
        return [x.item() for x in sample_IP]

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
        # SPECIAL CASE: num_generations=1 (e.g., validation set)
        # Cannot compute group-relative advantages with single sample per group
        # Instead, normalize across the entire batch
        if num_generations == 1:
            mean = rewards.mean()
            std = torch.clamp(rewards.std(), min=0.1)
            advantages = (rewards - mean) / std
            print(f"compute_group_advantages (batch-level): mean={mean:.4f}, std={std:.4f}")
            return advantages
        
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

    def step_tracin(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        val_queries: List[torch.LongTensor] = None,  # NOT USED - kept for API compatibility
        val_responses: List[torch.LongTensor] = None,  # NOT USED
        val_advantages: torch.FloatTensor = None,  # NOT USED
        val_logprobs: torch.FloatTensor = None,  # NOT USED
        val_masks: torch.FloatTensor = None,  # NOT USED
        val_scores: torch.FloatTensor = None,  # NOT USED
        gen_data_dir: str = None,
    ):
        """
        Run a GRPO optimization step with TracIn influence-based sample selection.
        
        IMPORTANT: Uses SAME-BATCH TracIn (like PPO's step_part_I)!
        The training batch is used as its own validation set.
        This is more appropriate for RL than using a separate validation set.
        
        Why same-batch TracIn works better:
        1. Training and validation are always in sync (same batch)
        2. Advantages are fresh and relevant to current model
        3. Selects samples that help the batch be self-consistent
        4. No stale validation data problem
        
        This method:
        1. Captures training gradients via hooks during forward+backward with ghost_mode
        2. Saves training gradient buffers
        3. Computes validation loss using SAME BATCH data
        4. Captures validation gradients via hooks
        5. Computes influence scores using efficient matrix operations
        6. Selects samples with positive influence
        7. Trains only on selected samples
        
        Args:
            queries: List of query tensors
            responses: List of response tensors
            scores: List of reward scores
            val_*: NOT USED - kept for API compatibility, ignored
            gen_data_dir: Directory to save generated data
            
        Returns:
            stats: Dictionary of training statistics
            ghost_ip: List of influence scores per sample
        """
        timing = {}
        t0 = time.time()
        
        bs = len(queries)
        
        # =====================
        # Part I: Prepare inputs and compute forward pass
        # =====================
        
        # Prepare model inputs
        t = time.time()
        model_inputs = self.prepare_model_inputs(queries, responses)
        model_inputs_names = list(model_inputs.keys())
        timing["time/grpo/prepare_inputs"] = time.time() - t
        
        # Put model in eval mode for gradient computation (disable dropout)
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        
        # Clear ghost buffers before starting
        self._clear_ghost_buffers()
        
        # Forward pass with ghost recording enabled (captures activations for training)
        t = time.time()
        self._record_ghost = True
        logprobs, masks = self.compute_logprobs(self.model, queries, responses, model_inputs)
        self._record_ghost = False
        
        # Get reference log probs
        with torch.no_grad():
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model), "disable_adapter"
            ):
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_logprobs, _ = self.compute_logprobs(self.model, queries, responses, model_inputs)
            elif self.ref_model is not None:
                ref_logprobs, _ = self.compute_logprobs(self.ref_model, queries, responses, model_inputs)
            else:
                ref_logprobs = logprobs.detach().clone()
        
        timing["time/grpo/forward_pass"] = time.time() - t
        
        # Convert scores to tensor
        scores_tensor = torch.tensor(scores, device=self.current_device, dtype=torch.float32)
        
        # Compute rewards with KL penalty
        t = time.time()
        rewards, non_score_reward = self.compute_rewards_with_kl(scores_tensor, logprobs.detach(), ref_logprobs, masks)
        timing["time/grpo/compute_rewards"] = time.time() - t
        
        # Compute group-relative advantages
        t = time.time()
        advantages = self.compute_group_advantages(scores_tensor, self.config.num_generations)
        advantages_expanded = advantages.unsqueeze(-1).expand_as(logprobs)
        timing["time/grpo/compute_advantages"] = time.time() - t
        
        # Prepare batch dict
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": logprobs.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "advantages_expanded": advantages_expanded,
        }
        batch_dict.update(model_inputs)
        
        # Store response lengths for consistent indexing
        max_response_len = logprobs.shape[1]
        
        # =====================
        # Part II: Capture TRAINING gradients via hooks (PPO-style)
        # =====================
        t = time.time()
        log_gpu_memory("Before training gradient capture", verbose=True)
        
        # Clear ghost buffers and start recording
        self._clear_ghost_buffers()
        self._record_ghost = True
        
        # Process training samples in batches with ghost_mode (no weight updates)
        for tracin_batch_start in range(0, bs, self.config.tracin_batch_size):
            tracin_batch_end = min(tracin_batch_start + self.config.tracin_batch_size, bs)
            tracin_batch_inds = np.arange(tracin_batch_start, tracin_batch_end)
            
            tracin_batch_dict = {
                "logprobs": batch_dict["logprobs"][tracin_batch_inds],
                "masks": batch_dict["masks"][tracin_batch_inds],
                "queries": [batch_dict["queries"][i] for i in tracin_batch_inds],
                "responses": [batch_dict["responses"][i] for i in tracin_batch_inds],
                "advantages": batch_dict["advantages"][tracin_batch_inds],
                "advantages_expanded": batch_dict["advantages_expanded"][tracin_batch_inds],
            }
            for k in model_inputs_names:
                tracin_batch_dict[k] = batch_dict[k][tracin_batch_inds]
            
            mb_model_inputs = {k: tracin_batch_dict[k] for k in model_inputs_names}
            
            # Forward pass (hooks capture activations)
            outputs = self.model(
                input_ids=mb_model_inputs["input_ids"],
                attention_mask=mb_model_inputs["attention_mask"],
            )
            logits = outputs.logits
            
            # Compute new log probs
            mb_logprobs_new = self._compute_logprobs_consistent(
                logits, tracin_batch_dict["queries"], tracin_batch_dict["responses"],
                mb_model_inputs, max_response_len
            )
            
            # Extract response logits for entropy
            response_logits = self._extract_response_logits_consistent(
                logits, tracin_batch_dict["queries"], tracin_batch_dict["responses"],
                max_response_len
            )
            
            # Compute loss for the batch
            pg_loss, _ = self.loss(
                tracin_batch_dict["logprobs"].detach(),
                response_logits,
                mb_logprobs_new,
                tracin_batch_dict["masks"].detach(),
                tracin_batch_dict["advantages_expanded"].detach(),
            )
            
            # Backward with ghost_mode (captures gradients via hooks but doesn't update weights)
            with ghost_mode(self.optimizer):
                self.accelerator.backward(pg_loss, retain_graph=False)
                self.optimizer.zero_grad()
        
        self._record_ghost = False
        
        # Save training gradient buffers
        self._save_train_buffers()
        
        timing["time/grpo/train_gradient_capture"] = time.time() - t
        log_gpu_memory("After training gradient capture", verbose=True)
        
        # =====================
        # Part III: Capture VALIDATION gradients using SAME BATCH (like PPO step_part_I)
        # =====================
        # KEY INSIGHT: Use the TRAINING BATCH as validation!
        # We need a FRESH forward pass because training backward consumed the graph.
        t = time.time()
        
        # Clear gradient buffers (keep activations from training in _train_* buffers)
        for buf in (self._xs, self._hs, self._gAs, self._gBs):
            for name in buf:
                buf[name] = []
        
        # FRESH forward pass for validation (needed because training backward consumed the graph)
        print(f"\n=== Validation Loss (Same-Batch TracIn) ===")
        print(f"Using TRAINING batch as validation (like PPO step_part_I)")
        print(f"Running fresh forward pass for validation gradient...")
        
        self._record_ghost = True
        val_outputs = self.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        )
        val_logits = val_outputs.logits
        self._record_ghost = False
        
        # Compute fresh logprobs from this forward pass
        val_logprobs = self._compute_logprobs_consistent(
            val_logits, queries, responses, model_inputs, max_response_len
        )
        
        print(f"val_logprobs shape: {val_logprobs.shape}, masks shape: {masks.shape}")
        print(f"advantages shape: {advantages.shape}")
        
        # Use seqloss-lastadv style loss on training batch (like PPO line 1210-1216)
        if self.config.val_loss_type in ['seqloss-lastadv', 'seqloss-reward']:
            seq_logprob = (val_logprobs.to(torch.float32) * masks.detach()).sum(dim=1)
            # Use batch advantages (already normalized, mean≈0)
            seq_score = advantages.detach()
            per_seq_loss = -seq_logprob * seq_score
            validation_loss = per_seq_loss.mean()
            print(f'Validation loss (same-batch seqloss): {validation_loss.item():.4f}')
            print(f'  seq_logprob range: [{seq_logprob.min():.4f}, {seq_logprob.max():.4f}]')
            print(f'  seq_score (advantages) range: [{seq_score.min():.4f}, {seq_score.max():.4f}]')
            print(f'  advantages mean: {advantages.mean():.4f}, std: {advantages.std():.4f}')
            
        elif self.config.val_loss_type == 'rough-orig':
            # Simple average loss (like PPO line 1206-1208)
            validation_loss = -torch.mean(advantages_expanded.detach() * val_logprobs.to(torch.float32) * masks.detach())
            print(f'Validation loss (rough-orig): {validation_loss.item():.4f}')
            
        elif self.config.val_loss_type == 'sample-level-orig':
            masked_term = advantages_expanded.detach() * val_logprobs.to(torch.float32) * masks.detach()
            per_sample_num = masks.sum(dim=1).clamp(min=1)
            per_sample_sum = masked_term.sum(dim=1)
            per_sample_loss = -per_sample_sum / per_sample_num
            validation_loss = per_sample_loss.mean()
            print(f'Validation loss (sample-level-orig): {validation_loss.item():.4f}')
            
        else:
            # Default: use rough-orig
            validation_loss = -torch.mean(advantages_expanded.detach() * val_logprobs.to(torch.float32) * masks.detach())
            print(f'Validation loss (default rough-orig): {validation_loss.item():.4f}')
        
        # Check for NaN
        if torch.isnan(validation_loss) or torch.isinf(validation_loss):
            print(f"ERROR: Validation loss is NaN/Inf! Using fallback loss.")
            validation_loss = -torch.mean(val_logprobs.to(torch.float32) * masks.detach())
        
        print(f"Validation loss final: {validation_loss.item():.4f}")
        print(f"================================\n")
        
        # Backward on validation loss (captures validation gradients via hooks)
        self._record_ghost = True
        self.accelerator.backward(validation_loss)
        self._record_ghost = False
        self.optimizer.zero_grad()
        
        # DEBUG: Check validation gradient norms
        if len(self._gAs) > 0:
            val_grad_norms = []
            for name in list(self._gAs.keys())[:3]:
                if len(self._gAs[name]) > 0:
                    gA_norm = torch.cat(self._gAs[name], dim=0).float().norm().item()
                    val_grad_norms.append(gA_norm)
            if val_grad_norms:
                print(f'Validation gradient norms (first 3 layers): {val_grad_norms}')
        
        # Free validation tensors
        del validation_loss, val_logprobs, val_outputs, val_logits
        
        timing["time/grpo/validation_gradient"] = time.time() - t
        log_gpu_memory("After validation gradient capture", verbose=True)
        
        # =====================
        # Part IV: Compute influence scores using efficient matrix operations
        # =====================
        t = time.time()
        
        # Compute ghost inner products using matrix operations (like PPO)
        ghost_ip = self.compute_ghost_inner_product_matrix_op()
        
        print(f"Ghost gradient inner products: {ghost_ip[:10]}...")
        if len(ghost_ip) > 0:
            valid_ips = [ip for ip in ghost_ip if not (math.isnan(ip) or math.isinf(ip))]
            if valid_ips:
                print(f"IP stats (valid): mean={np.mean(valid_ips):.6f}, std={np.std(valid_ips):.6f}, min={np.min(valid_ips):.6f}, max={np.max(valid_ips):.6f}")
            else:
                print("WARNING: All IPs are NaN/Inf!")
        
        # Check for NaN and replace with 0
        ghost_ip = [0.0 if (math.isnan(ip) or math.isinf(ip)) else ip for ip in ghost_ip]
        
        timing["time/grpo/tracin_calculation"] = time.time() - t
        
        # Clear ghost buffers to free memory
        self._clear_ghost_buffers()
        for buf in (self._train_xs, self._train_hs, self._train_gAs, self._train_gBs):
            for name in list(buf.keys()):
                buf[name] = []
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        log_gpu_memory("After TracIn calculation (buffers cleared)", verbose=True)
        
        # =====================
        # Part V: Select samples with positive influence and train
        # =====================
        t = time.time()
        
        # Analyze influence score distribution
        ghost_ip_array = np.array(ghost_ip)
        positive_count = np.sum(ghost_ip_array > 0)
        negative_count = np.sum(ghost_ip_array < 0)
        zero_count = np.sum(ghost_ip_array == 0)
        
        print(f'Influence score distribution:')
        print(f'  Positive: {positive_count} ({100*positive_count/bs:.1f}%)')
        print(f'  Negative: {negative_count} ({100*negative_count/bs:.1f}%)')
        print(f'  Zero: {zero_count} ({100*zero_count/bs:.1f}%)')
        print(f'  Mean: {np.mean(ghost_ip_array):.6f}, Std: {np.std(ghost_ip_array):.6f}')
        print(f'  Min: {np.min(ghost_ip_array):.6f}, Max: {np.max(ghost_ip_array):.6f}')
        
        # PPO-STYLE SAMPLE SELECTION: Select samples with positive influence
        # This matches PPO TracIn exactly
        selected_ids = np.where(ghost_ip_array > 0)[0]
        print(f'Number of selected samples (positive influence): {len(selected_ids)} / {bs}')
        
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
                "ghost_ip": ghost_ip,
                "selected_ids": selected_ids.tolist(),
            }, f'{gen_data_dir}/grpo_tracin_samples_seed-{self.config.seed}_{self.save_cnt}.pt')
            print(f'File saved to {gen_data_dir}/grpo_tracin_samples_seed-{self.config.seed}_{self.save_cnt}.pt')
            self.save_cnt += 1
        
        # PPO-STYLE TRAINING: Train ONLY on selected samples
        # This matches PPO TracIn exactly
        if len(selected_ids) > 0:
            sel_bs = len(selected_ids)
            all_stats = []
            self.model.train()
            
            print(f"Training on {sel_bs} selected samples (positive influence)")
            
            for epoch in range(self.config.grpo_epochs):
                # Shuffle selected indices
                b_inds = np.random.permutation(selected_ids)
                
                for backward_batch_start in range(0, sel_bs, self.config.backward_batch_size):
                    backward_batch_end = backward_batch_start + self.config.backward_batch_size
                    
                    # PPO-style: drop last batch if smaller than batch size
                    if backward_batch_end > sel_bs:
                        break
                    
                    backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]
                    
                    for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                        mini_batch_end = mini_batch_start + self.config.mini_batch_size
                        mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                        
                        if len(mini_batch_inds) == 0:
                            continue
                        
                        # Get mini-batch data
                        mb_logprobs_old = batch_dict["logprobs"][mini_batch_inds].detach()
                        mb_advantages = batch_dict["advantages_expanded"][mini_batch_inds].detach()
                        mb_masks = batch_dict["masks"][mini_batch_inds]
                        mb_queries = [batch_dict["queries"][i] for i in mini_batch_inds]
                        mb_responses = [batch_dict["responses"][i] for i in mini_batch_inds]
                        
                        mb_model_inputs = {
                            "input_ids": batch_dict["input_ids"][mini_batch_inds],
                            "attention_mask": batch_dict["attention_mask"][mini_batch_inds],
                        }
                        
                        # Forward pass
                        with self.accelerator.accumulate(self.model):
                            outputs = self.model(
                                input_ids=mb_model_inputs["input_ids"],
                                attention_mask=mb_model_inputs["attention_mask"],
                            )
                            logits = outputs.logits
                            
                            # Compute new log probs
                            mb_logprobs_new = self._compute_logprobs_consistent(
                                logits, mb_queries, mb_responses, mb_model_inputs, max_response_len
                            )
                            
                            # Extract response logits
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
                            
                            # Convert stats tensors to CPU/scalars before storing
                            stats_cpu = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for k, v in stats.items()}
                            all_stats.append(stats_cpu)
                            
                            # FREE mini-batch tensors immediately
                            del outputs, logits, mb_logprobs_new, response_logits, pg_loss
                            del mb_logprobs_old, mb_advantages, mb_masks, mb_model_inputs
            
            timing["time/grpo/optimization"] = time.time() - t
            
            # Memory cleanup after training loop
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory("After training loop", verbose=True)
        else:
            print("Warning: No samples with positive influence. Skipping training step.")
            all_stats = [{"loss/policy": 0.0, "loss/total": 0.0}]
            timing["time/grpo/optimization"] = time.time() - t
        
        # Aggregate stats (with safety check for empty list)
        # All stats are now scalars (converted to CPU in training loop)
        stats = {}
        if len(all_stats) == 0:
            print("WARNING: No training stats collected (training loop didn't execute). Using defaults.")
            stats = {"loss/policy": 0.0, "loss/total": 0.0}
        else:
            print(f"Aggregating stats from {len(all_stats)} training steps")
            for key in all_stats[0].keys():
                values = [s[key] for s in all_stats if key in s]
                if len(values) > 0:
                    stats[key] = np.mean(values)
        
        # Update KL controller
        kl = ((logprobs.detach() - ref_logprobs) * masks).sum(dim=-1).mean()
        self.kl_ctl.update(kl.item(), bs)
        
        # Record step stats
        stats.update(self.record_step_stats(
            kl_coef=self.kl_ctl.value,
            logprobs=logprobs.detach(),
            ref_logprobs=ref_logprobs,
            masks=masks,
            scores=scores_tensor,
            rewards=rewards,
        ))
        stats["tracin/num_selected"] = len(selected_ids)
        stats["tracin/selection_ratio"] = len(selected_ids) / bs
        stats["tracin/mean_ip"] = np.mean(ghost_ip)
        stats["tracin/std_ip"] = np.std(ghost_ip)
        stats["tracin/min_ip"] = np.min(ghost_ip)
        stats["tracin/max_ip"] = np.max(ghost_ip)
        stats["tracin/positive_ratio"] = np.sum(np.array(ghost_ip) > 0) / bs
        stats["tracin/negative_ratio"] = np.sum(np.array(ghost_ip) < 0) / bs
        stats.update(timing)
        stats["time/grpo/total"] = time.time() - t0
        
        # =====================
        # Final memory cleanup (SAFE - explicit try/except blocks)
        # =====================
        # Save return values first (before deleting ghost_ip)
        ghost_ip_copy = list(ghost_ip)
        
        # Delete batch_dict GPU tensors safely
        try:
            if isinstance(batch_dict, dict):
                for k in list(batch_dict.keys()):
                    if isinstance(batch_dict[k], torch.Tensor) and batch_dict[k].is_cuda:
                        del batch_dict[k]
                del batch_dict
        except (NameError, AttributeError, KeyError):
            pass
        
        # Delete GPU tensors explicitly (safe - wrapped in try/except)
        # These variables are defined in the function, so we can safely try to delete them
        try:
            del logprobs
        except NameError:
            pass
        try:
            del ref_logprobs
        except NameError:
            pass
        try:
            del masks
        except NameError:
            pass
        try:
            del rewards
        except NameError:
            pass
        try:
            del advantages
        except NameError:
            pass
        try:
            del advantages_expanded
        except NameError:
            pass
        try:
            del scores_tensor
        except NameError:
            pass
        try:
            del model_inputs
        except NameError:
            pass
        try:
            del trainable_params
        except NameError:
            pass
        try:
            del selected_ids
        except NameError:
            pass
        try:
            del ghost_ip
        except NameError:
            pass
        try:
            del all_stats
        except NameError:
            pass
        
        # Force Python garbage collection
        gc.collect()
        gc.collect()
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        log_gpu_memory("End of step_tracin (after cleanup)", verbose=True)
        
        return stats, ghost_ip_copy

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
