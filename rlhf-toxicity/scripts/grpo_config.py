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
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and target concatenate nested keys with separator."""

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


@dataclass
class GRPOConfig(object):
    """
    Configuration class for GRPOTrainer (Group Relative Policy Optimization)
    
    GRPO differs from PPO in that:
    - It does not use a value head (no critic network)
    - Advantages are computed as group-normalized rewards
    - Multiple generations per prompt are used to compute relative advantages
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of task to use - used only for tracking purposes"},
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of model to use - used only for tracking purposes"},
    )
    steps: Optional[int] = field(default=20000, metadata={"help": "Number of training steps"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "Adam learning rate"})
    
    # GRPO-specific: number of generations per prompt for group-relative advantages
    num_generations: Optional[int] = field(
        default=4,
        metadata={"help": "Number of generations per prompt for computing group-relative advantages"}
    )
    
    # KL penalty settings
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    kl_penalty: Optional[str] = field(
        default="kl",
        metadata={
            "help": "kl penalty options: 'kl': model_logp - ref_logp, 'abs': abs(kl), 'mse': mean squared error mse(kl)"
        },
    )
    target: Optional[float] = field(default=6, metadata={"help": "Target KL value for adaptive KL control"})
    horizon: Optional[float] = field(default=10000, metadata={"help": "Horizon for adaptive KL control"})
    
    # Clipping for policy gradient (no value clipping needed in GRPO)
    cliprange: Optional[float] = field(
        default=0.2, metadata={"help": "Range for clipping in GRPO policy gradient loss"}
    )
    
    # Batch sizes
    batch_size: Optional[int] = field(default=256, metadata={"help": "Number of samples per optimisation step"})
    forward_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples forward passed through model at a time"},
    )
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Number of samples optimized in each mini batch"}
    )
    
    # TracIn settings (for influence function analysis)
    tracin_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Number of samples used for TRACIN"}
    )
    tracin_val_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Number of samples used for TRACIN validation"}
    )
    val_loss_type: Optional[str] = field(
        default="rough-orig",
        metadata={
            "help": "Type of validation loss to use for TracIn computation"
        },
    )
    
    backward_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Number of samples optimized in an `optimizer.step()` call"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "The number of gradient accumulation steps"}
    )
    val_size: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Number of samples to use for validation"
        },
    )
    
    # GRPO training epochs (equivalent to ppo_epochs)
    grpo_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "Number of optimisation epochs per batch of samples"},
    )
    
    remove_unused_columns: Optional[bool] = field(
        default=True,
        metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"},
    )
    log_with: Optional[str] = field(
        default=None,
        metadata={
            "help": "Log with either 'wandb' or 'tensorboard'"
        },
    )
    tracker_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the tracker (e.g. wandb_project)"},
    )
    accelerator_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator"},
    )
    project_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator project config (e.g. `logging_dir`)"},
    )
    tracker_project_name: Optional[str] = field(
        default="trl", metadata={"help": "Name of project to use for tracking"}
    )
    max_grad_norm: Optional[float] = field(
        default=None, metadata={"help": "Maximum gradient norm for gradient clipping"}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "Seed value for random generations"})
    optimize_cuda_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Optimize CUDA cache for slightly more memory-efficient training"},
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "Whether to stop the GRPO optimization loop early if the KL is too high"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "Stop early if we exceed this value by over 50%"}
    )
    push_to_hub_if_best_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for pushing model to the hub during training (e.g. repo_id)"},
    )
    compare_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of steps between comparison of the current reward with the best seen so far"},
    )
    ratio_threshold: Optional[float] = field(
        default=10.0, metadata={"help": "Skip mini-batches with high ratios that can cause loss spikes"}
    )
    
    # Temperature for generation
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "Temperature for sampling during generation"}
    )

    def __post_init__(self):
        if self.forward_batch_size is not None:
            warnings.warn(
                "Note that using `forward_batch_size` is deprecated, use `mini_batch_size` instead."
            )
            self.mini_batch_size = self.forward_batch_size

        # check if wandb is installed
        if self.log_with == "wandb":
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Please install wandb to use wandb logging. You can do this by running `pip install wandb`."
                )

        self.total_grpo_epochs = int(np.ceil(self.steps / self.batch_size))
        assert self.kl_penalty in ["kl", "abs", "mse"]

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
