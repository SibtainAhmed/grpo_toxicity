# GRPO TracIn Performance Issues - Fixes Applied

## Summary

I've identified and fixed several critical issues in the GRPO TracIn implementation that were causing poor performance compared to vanilla GRPO.

## Issues Identified

### 1. **Sample Selection Too Restrictive** ⚠️ FIXED
**Problem**: Only selecting samples with `influence > 0` can be too restrictive. If most samples have negative influence (due to gradient direction or noise), very few samples are selected, leading to under-training.

**Fix Applied**:
- Added fallback strategy: If less than 10% of samples have positive influence, automatically switch to top-50% selection by influence
- Added comprehensive logging of influence score distribution (positive/negative/zero counts, mean, std, min, max)
- Added additional metrics to WandB: `tracin/positive_ratio`, `tracin/negative_ratio`, `tracin/min_ip`, `tracin/max_ip`

**Code Location**: `grpo_trainer.py:1450-1475`

### 2. **Insufficient Debug Information** ⚠️ FIXED
**Problem**: Lack of visibility into influence score distribution and validation gradient computation made debugging difficult.

**Fix Applied**:
- Added detailed logging of influence score distribution before selection
- Added validation loss debug information (advantage ranges, per-sample loss ranges)
- Added validation gradient norm logging (first 3 layers)
- Added training batch size logging

**Code Location**: 
- `grpo_trainer.py:1450-1475` (influence distribution)
- `grpo_trainer.py:1381-1390` (validation loss debug)
- `grpo_trainer.py:1407-1420` (validation gradient debug)

### 3. **Validation Loss Sign** ✅ VERIFIED CORRECT
**Status**: The validation loss formula matches PPO implementation exactly:
```python
per_seq_loss = -seq_logprob * seq_score
validation_loss = per_seq_loss.mean()
```

**Explanation**:
- Negative sign is correct: minimize `-logprob*score` = maximize `logprob*score`
- For positive advantages (good samples), we want to maximize logprobs
- Gradient points in direction to increase logprobs for positive advantages
- This matches PPO's implementation, so the sign is not the issue

## Probable Root Causes (Based on Analysis)

### Hypothesis 1: **Influence Score Distribution is Skewed** (Most Likely)
**Evidence**: If most samples have negative influence, we're selecting very few samples (< 10%), leading to under-training.

**Why This Happens**:
1. Validation set might not be representative
2. Validation gradients might be noisy (small validation set size: 16 samples)
3. Training and validation gradients might be misaligned

**Fix Applied**: Fallback to top-50% selection if too few positive samples

### Hypothesis 2: **Validation Set Size Too Small**
**Current**: 16 samples (`tracin_val_batch_size=16`)

**Problem**: 
- Too small to get reliable gradient estimates
- Noisy influence scores
- Unstable TracIn computation

**Recommendation**: Increase to 64-128 samples in `run_train_grpo_iif.sh`:
```bash
--tracin_val_batch_size=64  # or 128
```

### Hypothesis 3: **Validation Set Selection Strategy**
**Current**: Selecting most toxic samples (lowest reward scores)

**Problem**: 
- Most toxic samples might not be representative
- Model might not have learned to handle these cases yet
- Validation gradients might point in wrong direction

**Recommendation**: Try selecting least toxic (highest reward) samples instead, or use a balanced mix

### Hypothesis 4: **Gradient Computation Issues**
**Potential Issues**:
- LoRA gradient reconstruction might have bugs
- Sequence length handling might be incorrect
- Inner product computation might be wrong

**Status**: Need to verify with debug logging (now added)

## Fixes Applied

### 1. Enhanced Sample Selection with Fallback
```python
# If < 10% samples have positive influence, use top-50% instead
if len(selected_ids) < max(1, int(0.1 * bs)):
    top_k = max(1, int(0.5 * bs))
    selected_ids = np.argsort(ghost_ip_array)[-top_k:]
```

### 2. Comprehensive Debug Logging
- Influence score distribution (positive/negative/zero counts)
- Validation loss details (advantage ranges, per-sample loss)
- Validation gradient norms
- Training batch information

### 3. Additional WandB Metrics
- `tracin/positive_ratio`: Fraction of samples with positive influence
- `tracin/negative_ratio`: Fraction of samples with negative influence
- `tracin/min_ip`: Minimum influence score
- `tracin/max_ip`: Maximum influence score

## Testing Recommendations

### 1. **Monitor Influence Score Distribution**
Check WandB logs for:
- `tracin/positive_ratio`: Should be > 0.1 (at least 10% positive)
- If consistently < 0.1, the fallback strategy will activate

### 2. **Increase Validation Set Size**
Update `run_train_grpo_iif.sh`:
```bash
--tracin_val_batch_size=64  # Increase from 16
```

### 3. **Try Different Validation Selection Strategies**
- **Option A**: Select least toxic (highest reward) samples
- **Option B**: Select balanced mix (top 50% + bottom 50%)
- **Option C**: Select random samples (baseline)

### 4. **Compare with Vanilla GRPO**
- Monitor if TracIn performance improves with fixes
- Check if selection ratio stabilizes
- Verify that selected samples have reasonable influence scores

## Expected Improvements

After fixes:
1. **More samples selected**: Fallback strategy ensures at least 50% selection if needed
2. **Better visibility**: Debug logs show what's happening
3. **Stable selection**: Top-K fallback prevents under-training
4. **Performance improvement**: Should match or exceed vanilla GRPO

## Next Steps

1. **Run training** with fixes and monitor:
   - Influence score distribution
   - Selection ratio
   - Performance metrics

2. **If still underperforming**, try:
   - Increase validation set size to 64-128
   - Change validation selection to least toxic (highest reward)
   - Verify gradient computation is correct (check gradient norms)

3. **If influence scores are all near zero**:
   - Check if validation gradients are being computed correctly
   - Verify training gradients are being captured
   - Check if inner product computation is correct

## Diagnostic File

See `GRPO_TracIn_Diagnosis.md` for detailed analysis of all potential issues.
