# GRPO Standard Training Workflow

## Table of Contents
1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [High-Level Workflow](#high-level-workflow)
4. [Detailed Step-by-Step Process](#detailed-step-by-step-process)
5. [Code Structure & Key Components](#code-structure--key-components)
6. [Mathematical Formulation](#mathematical-formulation)

---

## Overview

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm for training language models to reduce toxicity. Unlike PPO, GRPO:
- **No value head**: Uses standard language model architecture
- **Group-based advantages**: Normalizes rewards within groups of responses from the same prompt
- **Simpler architecture**: Easier to implement and more memory-efficient

### What Problem Does It Solve?
- **Goal**: Train a language model to generate less toxic responses
- **Method**: Use a reward model (toxicity classifier) to score responses, then optimize the policy to maximize rewards while staying close to the reference model (KL penalty)

---

## Key Concepts

### 1. **Multiple Generations per Prompt**
- For each training prompt, we generate **N responses** (typically 4)
- These responses form a "group" for advantage computation
- Example: Prompt "Tell me about X" → 4 different responses → Group of 4

### 2. **Group-Relative Advantages**
- Instead of using raw rewards, we normalize within each group
- Formula: `advantage = (reward - group_mean) / group_std`
- This makes advantages relative to other responses from the same prompt
- Reduces variance and stabilizes training

### 3. **KL Penalty**
- Prevents the model from deviating too far from the reference model
- Ensures responses stay coherent and don't become gibberish
- Formula: `reward = raw_reward - kl_coef * KL(π_new || π_ref)`

### 4. **Policy Gradient Loss**
- Standard policy gradient: `loss = -log_prob * advantage`
- Clipped to prevent large policy updates: `loss = -min(ratio * advantage, clip(ratio) * advantage)`
- Where `ratio = exp(log_prob_new - log_prob_old)`

---

## High-Level Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO Standard Training Loop                    │
└─────────────────────────────────────────────────────────────────┘

For each training step:

1. DATA LOADING
   └─> Load batch of prompts (e.g., 64 prompts)

2. GENERATION PHASE
   └─> For each prompt:
       ├─> Generate 4 responses (num_generations=4)
       └─> Total: 64 prompts × 4 = 256 responses

3. REWARD COMPUTATION
   └─> For each response:
       ├─> Concatenate: prompt + response
       ├─> Feed to reward model (toxicity classifier)
       └─> Get reward score (lower = less toxic = better)

4. ADVANTAGE COMPUTATION
   └─> Group responses by prompt (64 groups of 4)
       ├─> For each group: normalize rewards
       └─> Result: 256 advantages (one per response)

5. POLICY UPDATE
   └─> For each mini-batch:
       ├─> Forward pass: compute new log probabilities
       ├─> Compute policy loss (with clipping)
       ├─> Backward pass: compute gradients
       └─> Update model weights

6. LOGGING & SAVING
   └─> Log metrics (rewards, KL, loss, etc.)
       └─> Save model checkpoint periodically
```

---

## Detailed Step-by-Step Process

### Step 1: Data Loading (`train_grpo.py` → `grpo_train_loop`)

**What happens:**
- Loads a batch of prompts from the toxicity dataset
- Each prompt is a potentially toxic text that needs a non-toxic response

**Code location:** `train_grpo.py:280-290`
```python
for batch in dataloader:
    question_tensors = batch["input_ids"]  # [batch_size, seq_len]
    queries_text = batch.get("query", ...)  # List of prompt strings
```

**Key point:** Batch size determines how many prompts we process per step (e.g., 64 prompts)

---

### Step 2: Generation Phase (`train_grpo.py` → `grpo_trainer.generate`)

**What happens:**
- For each prompt, generate **num_generations** responses (default: 4)
- Uses autoregressive generation with sampling (temperature, top-k, etc.)

**Code location:** `train_grpo.py:297-316`
```python
# Expand queries: repeat each query num_generations times
expanded_queries = []
for q_tensor in question_tensors:
    for _ in range(script_args.num_generations):  # Repeat 4 times
        expanded_queries.append(q_tensor)

# Batched generation - process all at once
all_responses = grpo_trainer.generate(
    expanded_queries,
    batch_size=gen_batch_size,  # Process in batches for efficiency
    **generation_kwargs
)
```

**Visual representation:**
```
Prompt 1 → [Response 1.1, Response 1.2, Response 1.3, Response 1.4]
Prompt 2 → [Response 2.1, Response 2.2, Response 2.3, Response 2.4]
...
Prompt 64 → [Response 64.1, Response 64.2, Response 64.3, Response 64.4]

Total: 256 responses (64 prompts × 4 generations)
```

**Key point:** Batched generation processes multiple queries in parallel for GPU efficiency

---

### Step 3: Reward Computation (`train_grpo.py` → `get_reward_scores`)

**What happens:**
- Concatenate each prompt with its response: `full_text = prompt + response`
- Feed to reward model (RoBERTa toxicity classifier)
- Get toxicity score (lower = less toxic = higher reward)

**Code location:** `train_grpo.py:206-239`
```python
def get_reward_scores(reward_model, reward_tokenizer, texts, device):
    # Tokenize full texts (prompt + response)
    inputs = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass through reward model
    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits.float()
        scores = logits[:, 0]  # Get toxicity score (first logit)
    
    return scores.cpu().tolist()
```

**Visual representation:**
```
Prompt: "Tell me about X"
Response: "X is a topic that..."
Full text: "Tell me about X X is a topic that..."

Reward Model (RoBERTa) → Score: 0.3 (low toxicity = good reward)
```

**Key point:** Reward model is frozen (no gradients), only used for scoring

---

### Step 4: Advantage Computation (`grpo_trainer.py` → `compute_group_advantages`)

**What happens:**
- Group responses by their original prompt
- For each group, compute mean and std of rewards
- Normalize: `advantage = (reward - group_mean) / group_std`
- Optionally normalize across entire batch (reduces variance)

**Code location:** `grpo_trainer.py:690-736`
```python
def compute_group_advantages(self, rewards, num_generations, normalize_batch=True):
    # Reshape: [256] → [64, 4] (64 prompts, 4 responses each)
    rewards_grouped = rewards.view(-1, num_generations)
    
    # Compute per-group statistics
    mean = rewards_grouped.mean(dim=1, keepdim=True)  # [64, 1]
    std = rewards_grouped.std(dim=1, keepdim=True)    # [64, 1]
    std = torch.clamp(std, min=0.1)  # Prevent division by zero
    
    # Normalize within each group
    advantages = (rewards_grouped - mean) / std  # [64, 4]
    
    # Flatten: [64, 4] → [256]
    advantages = advantages.view(-1)
    
    # Optional: Normalize across entire batch (like PPO)
    if normalize_batch:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages
```

**Visual representation:**
```
Group 1 (Prompt 1):
  Rewards: [0.3, 0.5, 0.2, 0.4]
  Mean: 0.35, Std: 0.12
  Advantages: [-0.42, 1.25, -1.25, 0.42]

Group 2 (Prompt 2):
  Rewards: [0.1, 0.8, 0.2, 0.3]
  Mean: 0.35, Std: 0.29
  Advantages: [-0.86, 1.55, -0.52, -0.17]

... (64 groups total)
```

**Key point:** Group-relative normalization makes advantages comparable across different prompts

---

### Step 5: Policy Update (`grpo_trainer.py` → `step`)

**What happens:**
- Prepare model inputs (concatenate queries + responses)
- Compute log probabilities for all responses
- Split into mini-batches for training
- For each mini-batch:
  - Forward pass: compute new log probabilities
  - Compute policy loss (with clipping)
  - Backward pass: compute gradients
  - Update weights

**Code location:** `grpo_trainer.py:964-1152`

#### 5.1: Prepare Inputs
```python
# Concatenate query + response for each sample
model_inputs = self.prepare_model_inputs(queries, responses)
# Result: input_ids [256, max_seq_len], attention_mask [256, max_seq_len]
```

#### 5.2: Compute Log Probabilities
```python
# Forward pass through model
logprobs, masks = self.compute_logprobs(self.model, queries, responses, model_inputs)
# logprobs: [256, max_response_len] - log probability of each token
# masks: [256, max_response_len] - 1 for valid tokens, 0 for padding

# Get reference log probs (from frozen reference model)
ref_logprobs, _ = self.compute_logprobs(self.ref_model, queries, responses, model_inputs)
```

#### 5.3: Compute Rewards with KL Penalty
```python
# KL divergence: measures how much new policy differs from reference
kl = (logprobs - ref_logprobs) * masks  # [256, max_response_len]
kl_per_sample = kl.sum(dim=-1)  # [256] - total KL per response

# Reward = raw_reward - kl_coef * KL
rewards = scores_tensor - self.kl_ctl.value * kl_per_sample
```

#### 5.4: Mini-Batch Training Loop
```python
# Split into mini-batches (e.g., 32 samples per mini-batch)
for mb_start in range(0, bs, mini_batch_size):
    mb_end = min(mb_start + mini_batch_size, bs)
    
    # Get mini-batch data
    mb_logprobs_old = logprobs[mb_start:mb_end]  # Cached from generation
    mb_advantages = advantages[mb_start:mb_end]
    mb_masks = masks[mb_start:mb_end]
    
    # Forward pass: compute NEW log probabilities
    outputs = self.model(mb_model_inputs)
    mb_logprobs_new = self._compute_logprobs_consistent(...)
    
    # Compute policy loss
    ratio = torch.exp(mb_logprobs_new - mb_logprobs_old)  # [mb_size, seq_len]
    pg_loss = -torch.min(
        ratio * mb_advantages_expanded,
        torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * mb_advantages_expanded
    ) * mb_masks
    
    # Backward pass
    self.accelerator.backward(pg_loss.mean())
    self.optimizer.step()
    self.optimizer.zero_grad()
```

**Visual representation:**
```
Full Batch (256 samples)
├─> Mini-batch 1 (samples 0-31)
│   ├─> Forward pass
│   ├─> Compute loss
│   ├─> Backward pass
│   └─> Update weights
├─> Mini-batch 2 (samples 32-63)
│   └─> ...
└─> ... (8 mini-batches total)
```

**Key point:** Mini-batching allows training on large batches without running out of memory

---

### Step 6: Logging & Saving

**What happens:**
- Aggregate statistics from all mini-batches
- Log to WandB (rewards, KL, loss, etc.)
- Save model checkpoint periodically

**Code location:** `train_grpo.py:340-375`
```python
# Log stats
stats.update(timing)
grpo_trainer.log_stats(stats, batch_log, scores)

# Save checkpoint
if epoch % script_args.save_freq == 0:
    grpo_trainer.save_model(f"{output_dir}/step_{epoch}")
```

---

## Code Structure & Key Components

### Main Files

1. **`train_grpo.py`**
   - **`grpo_train_loop()`**: Main training loop for standard GRPO
   - **`get_reward_scores()`**: Computes rewards using reward model
   - **`GRPOScriptArguments`**: Configuration dataclass

2. **`grpo_trainer.py`**
   - **`GRPOTrainer`**: Main trainer class
   - **`step()`**: Single training step (generation → reward → advantage → update)
   - **`compute_group_advantages()`**: Group-relative advantage computation
   - **`loss()`**: Policy gradient loss with clipping
   - **`generate()`**: Text generation

3. **`grpo_config.py`**
   - **`GRPOConfig`**: Configuration parameters (learning rate, batch size, etc.)

### Key Classes & Methods

#### `GRPOTrainer.step()`
**Purpose:** Execute one training step
**Inputs:**
- `queries`: List of prompt tensors
- `responses`: List of response tensors (already generated)
- `scores`: List of reward scores (already computed)

**What it does:**
1. Prepares model inputs
2. Computes log probabilities
3. Computes advantages (group-relative)
4. Trains on mini-batches
5. Returns statistics

**Key code sections:**
- Line 964-1000: Input preparation and log probability computation
- Line 1001-1020: Advantage computation
- Line 1021-1122: Mini-batch training loop
- Line 1123-1152: Statistics aggregation

#### `GRPOTrainer.compute_group_advantages()`
**Purpose:** Normalize rewards within groups
**Inputs:**
- `rewards`: Tensor of shape [batch_size * num_generations]
- `num_generations`: Number of responses per prompt

**What it does:**
1. Reshapes rewards into groups: [num_prompts, num_generations]
2. Computes mean and std per group
3. Normalizes: `(reward - mean) / std`
4. Optionally normalizes across entire batch

**Key code sections:**
- Line 715-716: Reshape into groups
- Line 719-723: Compute group statistics
- Line 726: Normalize within groups
- Line 733-734: Optional batch-level normalization

#### `GRPOTrainer.loss()`
**Purpose:** Compute policy gradient loss with clipping
**Inputs:**
- `old_logprobs`: Log probabilities from generation phase
- `new_logprobs`: Log probabilities from current forward pass
- `advantages`: Group-relative advantages
- `masks`: Attention masks

**What it does:**
1. Computes policy ratio: `exp(new_logprob - old_logprob)`
2. Clips ratio to prevent large updates
3. Computes loss: `-min(ratio * advantage, clip(ratio) * advantage)`

**Key code sections:**
- Line 850-870: Policy ratio computation
- Line 872-880: Clipped loss computation
- Line 882-890: Entropy bonus (encourages exploration)

---

## Mathematical Formulation

### 1. Reward Computation
```
r_raw(s, a) = RewardModel(prompt + response)
r(s, a) = r_raw(s, a) - β * KL(π_new || π_ref)
```
Where:
- `s` = prompt (state)
- `a` = response (action)
- `β` = KL coefficient (typically 0.04)
- `KL(π_new || π_ref)` = KL divergence between new and reference policy

### 2. Group-Relative Advantages
```
For group G_i (responses from prompt i):
  μ_i = mean({r(s_i, a_j) | j ∈ G_i})
  σ_i = std({r(s_i, a_j) | j ∈ G_i})
  A(s_i, a_j) = (r(s_i, a_j) - μ_i) / σ_i
```

### 3. Policy Gradient Loss
```
ratio = π_new(a|s) / π_old(a|s) = exp(log_prob_new - log_prob_old)
L_clip = -min(ratio * A, clip(ratio) * A)
L_entropy = -entropy(π_new)  # Encourages exploration
L_total = L_clip + c_entropy * L_entropy
```

### 4. Gradient Update
```
θ_new = θ_old + α * ∇_θ L_total
```
Where `α` is the learning rate.

---

## Summary

**GRPO Standard Training** is a straightforward reinforcement learning approach:
1. Generate multiple responses per prompt
2. Score them with a reward model
3. Compute group-relative advantages
4. Update policy to maximize advantages

**Key advantages:**
- Simple architecture (no value head)
- Memory efficient
- Stable training (group normalization reduces variance)

**Key differences from PPO:**
- No value function → simpler implementation
- Group-based advantages → better for multi-response scenarios
- Fewer hyperparameters → easier to tune
