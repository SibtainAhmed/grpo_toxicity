# GRPO TracIn Performance Issues - Diagnosis

## Problem Summary
GRPO TracIn is performing **worse than vanilla GRPO**, despite trying different validation selection strategies (random, highest-reward, lowest-reward). This suggests fundamental issues in the TracIn implementation.

## Critical Issues Identified

### Issue 1: **Validation Loss Sign is Backwards** ⚠️ CRITICAL

**Location**: `grpo_trainer.py:1381-1385`

**Current Code**:
```python
elif self.config.val_loss_type == 'seqloss-lastadv':
    seq_logprob = (val_logprobs_new.to(torch.float32) * val_masks.detach()).sum(dim=1)
    seq_score = val_advantages
    per_seq_loss = -seq_logprob * seq_score  # ← NEGATIVE SIGN!
    validation_loss = per_seq_loss.mean()
```

**Problem**:
- If `seq_score` (advantages) are **positive for good samples**, then `per_seq_loss = -seq_logprob * positive = negative`
- When we minimize loss via `backward()`, PyTorch computes `grad = ∂L/∂θ`
- The gradient points in the direction to **decrease** the loss
- But we want to **increase** logprobs for good samples (positive advantages)
- **The gradient direction is BACKWARDS!**

**Impact**:
- Validation gradients point in the wrong direction
- Influence scores have the **wrong sign**
- We select samples that **hurt** validation performance instead of help
- This explains why TracIn performs worse than vanilla GRPO!

**Fix**:
```python
# CORRECT: Remove the negative sign
per_seq_loss = seq_logprob * seq_score  # Positive for good samples
validation_loss = per_seq_loss.mean()
# OR if we want to minimize:
per_seq_loss = -seq_logprob * seq_score  # But then we need to maximize, not minimize
```

**But wait**: We need to check if PPO uses the same formula. If PPO works, then maybe the issue is elsewhere.

### Issue 2: **Influence Score Interpretation**

**Location**: `grpo_trainer.py:1451`

**Current Code**:
```python
selected_ids = np.where(np.array(ghost_ip) > 0)[0]
```

**Problem**:
- If validation loss gradient is backwards, then:
  - Training samples that **help** validation → **negative** influence
  - Training samples that **hurt** validation → **positive** influence
- We're selecting the **wrong samples**!

**Impact**:
- Only samples with positive influence are selected
- But if influence signs are wrong, we're selecting harmful samples
- This causes performance degradation

### Issue 3: **Sample Selection Too Restrictive**

**Current Behavior**:
- Only selects samples with `influence > 0`
- If most samples have negative influence (due to sign issue), very few samples are selected
- Leads to under-training

**Alternative Strategies** (from PPO code, commented out):
```python
# Option 1: Top 50% by influence
selected_ids = np.argsort(ghost_ip)[-int(len(ghost_ip) / 2):]

# Option 2: Bottom 50% by influence (if signs are wrong)
selected_ids = np.argsort(ghost_ip)[:int(len(ghost_ip) / 2)]

# Option 3: Random selection (baseline)
selected_ids = np.random.choice(np.arange(len(ghost_ip)), size=int(len(ghost_ip) / 2), replace=False)
```

### Issue 4: **Validation Set Size Too Small**

**Current**: 16 samples (from `tracin_val_batch_size=16`)

**Problem**:
- Too small to be representative
- Gradient estimates are noisy
- Influence scores are unreliable

**Recommendation**: Increase to 64-128 samples

### Issue 5: **Validation Advantages Normalization**

**Location**: `train_grpo.py:632`

**Current Code**:
```python
val_advantages = grpo_trainer.compute_group_advantages(val_scores_tensor, 1)
```

**Problem**:
- For `num_generations=1`, advantages are normalized across the batch
- But validation samples are independent (not grouped)
- Normalization might be causing issues

## Root Cause Analysis

### Hypothesis 1: Validation Loss Sign is Wrong
**Probability**: HIGH (90%)
- Explains why TracIn performs worse
- Explains why different validation selection strategies don't help
- Matches the symptom: selecting wrong samples

### Hypothesis 2: Influence Score Computation is Wrong
**Probability**: MEDIUM (50%)
- Gradient reconstruction might be incorrect
- LoRA gradient computation might have bugs
- Sequence length handling might be wrong

### Hypothesis 3: Sample Selection Strategy is Wrong
**Probability**: LOW (20%)
- PPO also uses `influence > 0`, so this is likely correct
- But if signs are wrong, this becomes critical

## Recommended Fixes

### Fix 1: Correct Validation Loss Sign (CRITICAL)

**Option A**: Remove negative sign (if we want to maximize good samples):
```python
per_seq_loss = seq_logprob * seq_score  # Positive for good samples
validation_loss = per_seq_loss.mean()
# Then we need to MAXIMIZE, not minimize
# But PyTorch minimizes by default, so we need to negate again
validation_loss = -per_seq_loss.mean()  # Minimize negative = maximize positive
```

**Option B**: Keep negative sign but check if PPO does the same:
- If PPO uses `-seq_logprob * seq_score`, then maybe advantages are defined differently
- Need to verify: Are GRPO advantages the same as PPO advantages?

### Fix 2: Add Debug Logging

Add comprehensive logging to verify:
1. Validation loss sign and magnitude
2. Validation gradient norms
3. Training gradient norms
4. Influence score distribution
5. Sample selection statistics

### Fix 3: Try Alternative Selection Strategies

Test different selection strategies:
1. Top-K by influence (instead of positive-only)
2. Bottom-K by influence (if signs are wrong)
3. Random selection (baseline)

### Fix 4: Increase Validation Set Size

Increase `tracin_val_batch_size` from 16 to 64-128

### Fix 5: Verify Advantage Computation

Check if validation advantages are computed correctly:
- Are they normalized correctly?
- Do they have the right sign?
- Are they comparable to training advantages?

## Testing Plan

1. **Fix validation loss sign** → Test if performance improves
2. **Add debug logging** → Verify influence scores are reasonable
3. **Try top-K selection** → Test if selection strategy matters
4. **Increase validation size** → Test if more samples help
5. **Compare with PPO** → Verify if PPO has the same issue

## Expected Outcomes

After fixes:
- TracIn should perform **better** than vanilla GRPO (or at least similar)
- Influence scores should have correct signs
- Sample selection should pick helpful samples
- Performance should be stable and improving
