# GRPO TracIn Training Workflow

## Table of Contents
1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [High-Level Workflow](#high-level-workflow)
4. [Detailed Step-by-Step Process](#detailed-step-by-step-process)
5. [Code Structure & Key Components](#code-structure--key-components)
6. [Mathematical Formulation](#mathematical-formulation)
7. [Comparison with Standard GRPO](#comparison-with-standard-grpo)

---

## Overview

**GRPO TracIn Training** extends standard GRPO with **influence-based sample selection**. Instead of training on all generated responses, we:
1. Compute which training samples would most improve validation performance
2. Select only samples with positive influence
3. Train only on selected samples

### What Problem Does It Solve?
- **Goal**: Train more efficiently by focusing on samples that help validation performance
- **Method**: Use gradient-based influence scores (TracIn) to identify helpful training samples
- **Benefit**: Better generalization, faster convergence, reduced overfitting

### TracIn (Training Data Influence)
TracIn measures how much a training sample influences the model's performance on validation data by computing the inner product of gradients:
```
Influence(train_sample, val_sample) = <∇_θ L_train, ∇_θ L_val>
```
- Positive influence → training sample helps validation performance
- Negative influence → training sample hurts validation performance

---

## Key Concepts

### 1. **Validation Set**
- Separate set of prompts used to measure model performance
- Not used for training, only for evaluation
- Used to compute validation gradients for TracIn

### 2. **Gradient-Based Influence**
- For each training sample, compute its gradient: `∇_θ L_train`
- For validation set, compute aggregated gradient: `∇_θ L_val`
- Influence = inner product: `Influence = <∇_θ L_train, ∇_θ L_val>`
- Positive influence → sample helps validation → select it
- Negative influence → sample hurts validation → skip it

### 3. **Hook-Based Gradient Capture**
- Uses PyTorch hooks to capture gradients during forward/backward pass
- More memory-efficient than `torch.autograd.grad()` (reuses computational graph)
- Captures gradients for LoRA layers (low-rank adaptation layers)

### 4. **Ghost Mode**
- Context manager that runs backward pass without updating weights
- Allows gradient computation without modifying model parameters
- Essential for TracIn (we need gradients, not weight updates)

---

## High-Level Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                  GRPO TracIn Training Loop                        │
└─────────────────────────────────────────────────────────────────┘

For each training step:

1. DATA LOADING
   └─> Load batch of prompts (e.g., 64 prompts)

2. GENERATION PHASE
   ├─> Generate training responses (64 prompts × 4 = 256 responses)
   └─> Generate validation responses (for TracIn computation)

3. REWARD COMPUTATION
   ├─> Compute rewards for training responses
   └─> Compute rewards for validation responses

4. TRACIN COMPUTATION (NEW!)
   ├─> Part I: Capture training gradients via hooks
   │   └─> For each training sample, compute gradient
   │       └─> Store in buffers (_train_xs, _train_hs, _train_gAs, _train_gBs)
   │
   ├─> Part II: Capture validation gradients via hooks
   │   └─> For validation set, compute aggregated gradient
   │       └─> Store in buffers (_xs, _hs, _gAs, _gBs)
   │
   ├─> Part III: Compute influence scores
   │   └─> For each training sample:
   │       └─> Influence = <grad_train, grad_val>
   │
   └─> Part IV: Select samples
       └─> Keep only samples with positive influence

5. POLICY UPDATE (ONLY ON SELECTED SAMPLES)
   └─> Train only on selected samples (e.g., 128 out of 256)

6. LOGGING & SAVING
   └─> Log metrics (rewards, KL, loss, influence scores, selection ratio)
       └─> Save model checkpoint periodically
```

---

## Detailed Step-by-Step Process

### Step 1-3: Data Loading, Generation, Reward Computation

**Same as Standard GRPO** (see Standard GRPO document)

**Additional step for TracIn:**
- Generate validation responses once (or regenerate periodically)
- Compute validation rewards and advantages

**Code location:** `train_grpo.py:424-552`
```python
# Pre-generate validation responses
val_sample_size = min(script_args.tracin_val_batch_size, len(val_question_tensors))
val_sample_indices = torch.randperm(len(val_question_tensors))[:val_sample_size]
val_queries_batch = [val_question_tensors[idx].to(device) for idx in val_sample_indices]

# Generate validation responses
with torch.no_grad():
    val_responses_batch = grpo_trainer.generate(
        val_queries_batch,
        batch_size=gen_batch_size,
        **generation_kwargs
    )

# Compute validation rewards and advantages
val_scores = get_reward_scores(reward_model, reward_tokenizer, val_full_texts, device)
val_advantages = grpo_trainer.compute_group_advantages(
    torch.tensor(val_scores), num_generations=1
)
```

---

### Step 4: TracIn Computation (`grpo_trainer.py` → `step_tracin`)

This is the **core difference** from standard GRPO. Let's break it down:

#### Part I: Capture Training Gradients

**What happens:**
- For each training sample, run forward + backward pass
- Use hooks to capture activations and gradients
- Store in buffers for later use

**Code location:** `grpo_trainer.py:1250-1317`

```python
# Clear ghost buffers
self._clear_ghost_buffers()
self._record_ghost = True  # Enable hook recording

# Process training samples in batches
for tracin_batch_start in range(0, bs, tracin_batch_size):
    # Forward pass (hooks capture activations: _xs, _hs)
    outputs = self.model(mb_model_inputs)
    logits = outputs.logits
    
    # Compute loss
    pg_loss, _ = self.loss(...)
    
    # Backward with ghost_mode (captures gradients: _gAs, _gBs, but doesn't update weights)
    with ghost_mode(self.optimizer):
        self.accelerator.backward(pg_loss)
        self.optimizer.zero_grad()  # Clear gradients (but hooks already captured them)

self._record_ghost = False
self._save_train_buffers()  # Copy to _train_xs, _train_hs, _train_gAs, _train_gBs
```

**Visual representation:**
```
Training Sample 1:
  Forward → Hooks capture: x₁, h₁ (activations)
  Backward → Hooks capture: gA₁, gB₁ (gradients)
  Store in: _train_xs, _train_hs, _train_gAs, _train_gBs

Training Sample 2:
  Forward → Hooks capture: x₂, h₂
  Backward → Hooks capture: gA₂, gB₂
  Store in: _train_xs, _train_hs, _train_gAs, _train_gBs

... (256 training samples)
```

**Key point:** `ghost_mode()` prevents weight updates, but gradients are still computed and captured by hooks

---

#### Part II: Capture Validation Gradients

**What happens:**
- Clear training buffers (keep _train_* buffers)
- Run forward + backward on validation set
- Capture validation gradients

**Code location:** `grpo_trainer.py:1319-1380`

```python
# Clear ALL ghost buffers (training data is saved in _train_* buffers)
self._clear_ghost_buffers()

# Prepare validation inputs
val_model_inputs = self.prepare_model_inputs(val_queries, val_responses)

# Forward pass (hooks capture validation activations)
self._record_ghost = True
val_outputs = self.model(val_model_inputs)
val_logits = val_outputs.logits

# Compute validation loss
val_logprobs_new = self._compute_logprobs_consistent(...)
validation_loss = compute_validation_loss(val_logprobs_new, val_advantages, val_masks)

# Backward (hooks capture validation gradients)
self.accelerator.backward(validation_loss)
self.optimizer.zero_grad()
self._record_ghost = False
```

**Visual representation:**
```
Validation Set (4 samples):
  Forward → Hooks capture: x_val, h_val (validation activations)
  Backward → Hooks capture: gA_val, gB_val (validation gradients)
  Store in: _xs, _hs, _gAs, _gBs (overwrites training data)
```

**Key point:** Validation gradients are aggregated (sum over validation samples) to get a single "validation direction"

---

#### Part III: Compute Influence Scores

**What happens:**
- For each training sample, compute gradient inner product with validation gradient
- This measures how much the training sample influences validation performance

**Code location:** `grpo_trainer.py:1382-1416` → `compute_ghost_inner_product_matrix_op()`

```python
def compute_ghost_inner_product_matrix_op(self):
    # For each LoRA layer:
    for name in self._train_xs:
        # Training data: [N_train, S_train, dim]
        train_X = torch.cat(self._train_xs[name], dim=0)    # Activations
        train_GAt = torch.cat(self._train_gAs[name], dim=0) # Gradients
        
        # Validation data: [N_val, S_val, dim]
        val_X = torch.cat(self._xs[name], dim=0)
        val_GAt = torch.cat(self._gAs[name], dim=0)
        
        # Reconstruct LoRA gradients (sequence-length independent)
        # grad_W_A = sum_s(g_A[s].T @ x[s]) → [r, d_in]
        train_grad_A = torch.einsum('nsr,nsd->nrd', train_GAt.float(), train_X.float())
        val_grad_A = (val_GAt.float().reshape(-1, r, 1) @ val_X.float().reshape(-1, 1, d_in)).sum(dim=0)
        
        # Compute inner product per training sample
        train_grad_A_flat = train_grad_A.view(N_train, -1)
        val_grad_A_flat = val_grad_A.flatten()
        
        ip_A = (train_grad_A_flat * val_grad_A_flat.unsqueeze(0)).sum(dim=1)
        
        # Sum over all layers
        sample_IP[:N_train] += ip_A + ip_B
```

**Visual representation:**
```
For Training Sample i:
  grad_train[i] = [grad_layer1[i], grad_layer2[i], ..., grad_layerN[i]]
  grad_val = [grad_val_layer1, grad_val_layer2, ..., grad_val_layerN]
  
  Influence[i] = <grad_train[i], grad_val>
               = sum_over_layers( <grad_train[i]_layer, grad_val_layer> )
```

**Key point:** Uses efficient matrix operations (einsum) to compute gradients for LoRA layers, making it sequence-length independent

---

#### Part IV: Select Samples

**What happens:**
- Filter training samples: keep only those with positive influence
- This is the "sample selection" step

**Code location:** `grpo_trainer.py:1418-1450`

```python
# Compute influence scores
ghost_ip = self.compute_ghost_inner_product_matrix_op()
# Result: [256] - one influence score per training sample

# Select samples with positive influence
selected_ids = np.where(np.array(ghost_ip) > 0)[0]
# Result: e.g., [0, 2, 5, 7, ...] - indices of selected samples

print(f'Number of selected samples (positive influence): {len(selected_ids)} / {bs}')
# Example output: "Number of selected samples: 128 / 256"
```

**Visual representation:**
```
Training Samples (256):
  Sample 0: Influence = +0.5  → SELECT ✓
  Sample 1: Influence = -0.2  → SKIP ✗
  Sample 2: Influence = +0.3  → SELECT ✓
  Sample 3: Influence = -0.1  → SKIP ✗
  ...

Selected: 128 samples (50% selection rate)
```

**Key point:** Only samples that help validation performance are used for training

---

### Step 5: Policy Update (Only on Selected Samples)

**What happens:**
- Same as standard GRPO, but only on selected samples
- Split selected samples into mini-batches
- Train on each mini-batch

**Code location:** `grpo_trainer.py:1452-1520`

```python
# Filter batch_dict to only selected samples
selected_batch_dict = {
    "queries": [batch_dict["queries"][i] for i in selected_ids],
    "responses": [batch_dict["responses"][i] for i in selected_ids],
    "logprobs": batch_dict["logprobs"][selected_ids],
    "advantages": batch_dict["advantages"][selected_ids],
    ...
}

# Train on selected samples (same as standard GRPO step)
for mb_start in range(0, len(selected_ids), mini_batch_size):
    # ... standard GRPO training loop ...
    pg_loss, stats = self.loss(...)
    self.accelerator.backward(pg_loss)
    self.optimizer.step()
```

**Visual representation:**
```
Selected Samples (128):
├─> Mini-batch 1 (samples 0-31)
│   ├─> Forward pass
│   ├─> Compute loss
│   ├─> Backward pass
│   └─> Update weights
├─> Mini-batch 2 (samples 32-63)
│   └─> ...
└─> ... (4 mini-batches total)

Skipped Samples (128):
└─> Not used for training
```

**Key point:** Training is more efficient because we only update on helpful samples

---

### Step 6: Logging & Saving

**What happens:**
- Log standard metrics (rewards, KL, loss)
- **Additional TracIn metrics:**
  - Mean influence score
  - Number of selected samples
  - Selection ratio

**Code location:** `train_grpo.py:594-620`

```python
# Log stats
stats.update(timing)
stats["tracin/mean_ip"] = np.mean(ghost_ip)
stats["tracin/num_selected"] = len(selected_ids)
stats["tracin/selection_ratio"] = len(selected_ids) / bs

grpo_trainer.log_stats(stats, batch_log, scores)
```

---

## Code Structure & Key Components

### Main Files

1. **`train_grpo.py`**
   - **`grpo_train_loop_with_validation()`**: Main training loop for TracIn GRPO
   - **`get_reward_scores()`**: Computes rewards (same as standard)

2. **`grpo_trainer.py`**
   - **`step_tracin()`**: TracIn training step (replaces `step()`)
   - **`compute_ghost_inner_product_matrix_op()`**: Computes influence scores
   - **`_install_ghost_hooks()`**: Installs hooks on LoRA layers
   - **`_save_train_buffers()`**: Saves training gradients
   - **`_clear_ghost_buffers()`**: Clears gradient buffers
   - **`ghost_mode()`**: Context manager for gradient-only backward pass

### Key Classes & Methods

#### `GRPOTrainer.step_tracin()`
**Purpose:** Execute one TracIn training step
**Inputs:**
- `queries`, `responses`, `scores`: Training data
- `val_queries`, `val_responses`, `val_advantages`, `val_logprobs`, `val_masks`: Validation data

**What it does:**
1. Captures training gradients via hooks
2. Captures validation gradients via hooks
3. Computes influence scores
4. Selects samples with positive influence
5. Trains only on selected samples

**Key code sections:**
- Line 1250-1317: Part I - Capture training gradients
- Line 1319-1380: Part II - Capture validation gradients
- Line 1382-1416: Part III - Compute influence scores
- Line 1418-1450: Part IV - Select samples
- Line 1452-1520: Part V - Train on selected samples

#### `GRPOTrainer._install_ghost_hooks()`
**Purpose:** Install forward/backward hooks on LoRA layers
**What it does:**
- Finds all LoRA layers in the model
- Registers forward hook on `lora_A`: captures input `x` and output `h`
- Registers backward hook on `lora_B`: captures gradients `gA` and `gB`

**Code location:** `grpo_trainer.py:359-409`
```python
def _install_ghost_hooks(self):
    for name, module in self.model.named_modules():
        if isinstance(module, LoraLayer):
            # Forward hook: capture x (input) and h (output of lora_A)
            def fwd_A(mod, inp, out, nm=name):
                if self._record_ghost:
                    x, = inp  # [B, S, d_in]
                    h = out   # [B, S, r]
                    self._xs[nm].append(x.detach().clone())
                    self._hs[nm].append(h.detach().clone())
            
            module.lora_A.default.register_forward_hook(fwd_A)
            
            # Backward hook: capture gA and gB (gradients)
            def bwd_B(mod, grad_inp, grad_out, nm=name):
                if self._record_ghost:
                    g_h = grad_inp[0]  # [B, S, r] - gradient w.r.t. lora_A output
                    g_o = grad_out[0]  # [B, S, d_out] - gradient w.r.t. lora_B output
                    self._gAs[nm].append(g_h.detach().clone())
                    self._gBs[nm].append(g_o.detach().clone())
            
            module.lora_B.default.register_full_backward_hook(bwd_B)
```

**Key point:** Hooks only capture when `_record_ghost = True`, allowing selective gradient capture

#### `GRPOTrainer.compute_ghost_inner_product_matrix_op()`
**Purpose:** Compute influence scores using efficient matrix operations
**What it does:**
1. Reconstructs LoRA gradients from captured activations and gradients
2. Aggregates validation gradients
3. Computes inner product per training sample

**Code location:** `grpo_trainer.py:452-511`
```python
def compute_ghost_inner_product_matrix_op(self):
    # For each LoRA layer:
    for name in self._train_xs:
        # Training: [N_train, S_train, dim]
        train_X = torch.cat(self._train_xs[name], dim=0)
        train_GAt = torch.cat(self._train_gAs[name], dim=0)
        
        # Validation: [N_val, S_val, dim]
        val_X = torch.cat(self._xs[name], dim=0)
        val_GAt = torch.cat(self._gAs[name], dim=0)
        
        # Reconstruct gradients (sequence-length independent)
        train_grad_A = torch.einsum('nsr,nsd->nrd', train_GAt.float(), train_X.float())
        val_grad_A = (val_GAt.float().reshape(-1, r, 1) @ val_X.float().reshape(-1, 1, d_in)).sum(dim=0)
        
        # Inner product
        ip_A = (train_grad_A_flat * val_grad_A_flat.unsqueeze(0)).sum(dim=1)
        
        sample_IP += ip_A + ip_B
```

**Key point:** Uses einsum for efficient gradient reconstruction, making it sequence-length independent

---

## Mathematical Formulation

### 1. TracIn Influence Score

For a training sample `(x_train, y_train)` and validation set `{(x_val, y_val)}`:

```
Influence(train, val) = <∇_θ L_train(x_train, y_train), ∇_θ L_val(x_val, y_val)>
```

Where:
- `L_train` = training loss (policy gradient loss)
- `L_val` = validation loss (sequence-level or sample-level)
- `∇_θ` = gradient w.r.t. model parameters

### 2. LoRA Gradient Reconstruction

For LoRA layer with weight `W_A = [r, d_in]`:

```
grad_W_A = sum_over_sequence( g_A[s].T @ x[s] )
```

Where:
- `g_A[s]` = gradient w.r.t. LoRA output at position `s` (from backward hook)
- `x[s]` = input activation at position `s` (from forward hook)

### 3. Influence Computation

```
For each training sample i:
  grad_train[i] = [grad_layer1[i], grad_layer2[i], ..., grad_layerN[i]]
  
For validation set:
  grad_val = sum_over_val_samples( [grad_val_layer1, grad_val_layer2, ..., grad_val_layerN] )
  
Influence[i] = sum_over_layers( <grad_train[i]_layer, grad_val_layer> )
```

### 4. Sample Selection

```
selected_samples = {i | Influence[i] > 0}
```

Only samples with positive influence are used for training.

---

## Comparison with Standard GRPO

### Similarities

| Aspect | Standard GRPO | TracIn GRPO |
|--------|---------------|-------------|
| Generation | Generate 4 responses per prompt | Same |
| Reward computation | Use reward model to score responses | Same |
| Advantage computation | Group-relative normalization | Same |
| Policy update | Policy gradient with clipping | Same (but only on selected samples) |
| Architecture | No value head, LoRA fine-tuning | Same |

### Differences

| Aspect | Standard GRPO | TracIn GRPO |
|--------|---------------|-------------|
| **Sample selection** | Train on ALL samples | Train only on SELECTED samples |
| **Gradient computation** | Only for training | For both training AND validation |
| **Memory usage** | Lower | Higher (gradient buffers) |
| **Training time** | Faster per step | Slower per step (gradient computation overhead) |
| **Convergence** | Standard | Potentially faster (focus on helpful samples) |
| **Generalization** | Standard | Potentially better (validation-guided selection) |

### When to Use Each

**Standard GRPO:**
- Faster training
- Lower memory usage
- Simpler implementation
- Good for initial experiments

**TracIn GRPO:**
- Better generalization
- Faster convergence (fewer steps needed)
- Validation-guided training
- Good for final model training

---

## Summary

**GRPO TracIn Training** extends standard GRPO with influence-based sample selection:

1. **Generate** training and validation responses
2. **Capture** gradients for both training and validation sets (via hooks)
3. **Compute** influence scores (gradient inner products)
4. **Select** samples with positive influence
5. **Train** only on selected samples

**Key advantages:**
- Better generalization (validation-guided)
- Faster convergence (focus on helpful samples)
- More efficient training (skip harmful samples)

**Key challenges:**
- Higher memory usage (gradient buffers)
- Slower per-step (gradient computation overhead)
- More complex implementation (hooks, buffer management)

**Key innovation:**
- Hook-based gradient capture (memory-efficient)
- Sequence-length independent gradient reconstruction (efficient matrix operations)
- Validation-guided sample selection (better generalization)
