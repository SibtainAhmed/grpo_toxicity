# GENERAL CODE FOR RLHF TRAINING ON OUR DIFFERENT SETTINGS

import os
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import  PPOTrainer, set_seed
import wandb


from rlhfutils.rl_utils import (
    ScriptArguments as _BaseScriptArguments,
    load_models,
    train_loop,
    train_loop_one_step,
    train_loop_with_validation,
)


# Extend ScriptArguments with evaluation fields
# (the installed rl_utils.py may not have these)
@dataclass
class ScriptArguments(_BaseScriptArguments):
    eval_freq: Optional[int] = field(default=0, metadata={"help": "evaluate every N steps (0 to disable)"})
    eval_num_samples: Optional[int] = field(default=256, metadata={"help": "number of test prompts to evaluate on"})
    eval_toxicity_model: Optional[str] = field(default="s-nlp/roberta_toxicity_classifier", metadata={"help": "different toxicity detector for evaluation"})


# ============================================================================
# Evaluation functions (self-contained, no dependency on installed rl_utils)
# ============================================================================
def load_eval_toxicity_model(model_name, device):
    """Load a DIFFERENT toxicity classifier for evaluation."""
    from transformers import pipeline as hf_pipeline
    print(f"\n=== Loading Evaluation Toxicity Model ===")
    print(f"  Model: {model_name}")
    print(f"  (Different from training reward model!)")
    eval_classifier = hf_pipeline(
        "text-classification", model=model_name,
        device=device if isinstance(device, int) else 0,
        batch_size=64, truncation=True, max_length=512,
    )
    print(f"  Eval toxicity model loaded successfully!")
    return eval_classifier


def evaluate_toxicity_on_test_set(ppo_trainer, tokenizer, test_prompts, eval_classifier,
                                   device, num_samples=256, max_new_tokens=20, generation_kwargs=None):
    """Evaluate model toxicity on a FIXED held-out test set using a different toxicity detector.
    NOTE: test_prompts should be PRE-SAMPLED once at startup and reused across all eval steps.
    This eliminates sampling variance and makes eval/toxicity_mean stable and comparable."""
    num_eval = min(num_samples, len(test_prompts))
    # Use ALL pre-sampled prompts (no re-sampling — fixed set for stable comparisons)
    eval_prompts = test_prompts[:num_eval]
    if generation_kwargs is None:
        generation_kwargs = {
            "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id, "max_new_tokens": max_new_tokens, "temperature": 1.0,
        }
    all_texts = []
    ppo_trainer.model.eval()
    with torch.no_grad():
        for bs in range(0, num_eval, 32):
            be = min(bs + 32, num_eval)
            batch = eval_prompts[bs:be]
            gen_kw = {k: v for k, v in generation_kwargs.items()}
            if "max_new_tokens" not in gen_kw:
                gen_kw["max_new_tokens"] = max_new_tokens
            # Generate responses — PPOTrainer.generate returns full sequences (prompt+response)
            full_resp = ppo_trainer.generate(batch, **gen_kw)
            # Decode prompts and full responses separately to build full text
            prompt_lens = [len(p) for p in batch]
            for i, full_seq in enumerate(full_resp):
                # Strip prompt tokens to get response-only, then concatenate as text
                resp_tokens = full_seq[prompt_lens[i]:]
                prompt_text = tokenizer.decode(batch[i], skip_special_tokens=True)
                resp_text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
                all_texts.append(prompt_text + resp_text)
            del full_resp
    ppo_trainer.model.train()
    if not all_texts:
        return {"eval/toxicity_mean": 0.0, "eval/toxicity_std": 0.0, "eval/num_samples": 0}
    toxicity_scores = []
    for bs in range(0, len(all_texts), 64):
        results = eval_classifier(all_texts[bs:bs + 64])
        for result in results:
            if isinstance(result, list):
                sc = 0.0
                for ld in result:
                    lb = ld.get("label", "").lower()
                    if "toxic" in lb or lb == "label_1":
                        sc = ld["score"]; break
                toxicity_scores.append(sc)
            elif isinstance(result, dict):
                lb = result.get("label", "").lower(); sc = result["score"]
                toxicity_scores.append(sc if ("toxic" in lb or lb == "label_1") else 1.0 - sc)
    ta = np.array(toxicity_scores)
    return {
        "eval/toxicity_mean": float(np.mean(ta)), "eval/toxicity_std": float(np.std(ta)),
        "eval/toxicity_max": float(np.max(ta)), "eval/toxicity_min": float(np.min(ta)),
        "eval/toxic_frac": float(np.mean(ta > 0.5)), "eval/num_samples": num_eval,
    }


from rlhfutils.data import (
    build_wgpt_promptdata,
    build_rlcd_promptdata,
    build_stack_promptdata,
    build_apf_promptdata,
    build_ultra_promptdata,
    build_custom_promptdata, 
    build_imdb_promptdata,
    build_toxicity_promptdata,
    collator,
    qaform,
    anscat
)

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
tqdm.pandas()

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

wandb.init(project=script_args.wandb_project, name=script_args.run_name, config=script_args)

set_seed(script_args.seed)

if script_args.output_dir[-1]!="/":
    script_args.output_dir = script_args.output_dir+"/"

print("over here")
# NOTE special case if using an api endpoint
if "http" in script_args.reward_model_name:
    config, tokenizer, model, optimizer = load_models(script_args, "ppo")
    reward_model = None
elif "function" in script_args.reward_model_name:
    config, tokenizer, model, optimizer = load_models(script_args, "ppo")
    reward_model = "function"
else:
    # NOTE handle loading everything in, since hyperparams are same for every setting more or less
    config, tokenizer, model, optimizer, reward_model, reward_tokenizer = load_models(script_args)

print("loaded models")


print('======= dataset', script_args.dataset_name)
rmformat = qaform
if "wgpt" == script_args.dataset_name:
    dataset = build_wgpt_promptdata(tokenizer)
    # TODO the ones below this
elif "rlcd" in script_args.dataset_name:
    dataset = build_rlcd_promptdata(tokenizer, script_args.dataset_name)
    rmformat = anscat  # NOTE RLCD RM has a different prompt template depending on the model, this is a bit ad-hoc
elif "stack" == script_args.dataset_name:
    dataset = build_stack_promptdata(tokenizer)
    rmformat = anscat
elif "apfarm" == script_args.dataset_name:
    dataset = build_apf_promptdata(tokenizer)
    rmformat = anscat
# TODO fix ultrachat datset issue
elif "ultra" == script_args.dataset_name:
    print("NOTE we're not using custom data, we're using default ultafeedback here")
    # TODO maybe unify original prompt format? 
    dataset = build_ultra_promptdata(tokenizer)
elif "imdb" in script_args.dataset_name:
    dataset = build_imdb_promptdata(tokenizer)
    if script_args.with_validation:
        valid_dataset = build_imdb_promptdata(tokenizer, split='test', num_samples=script_args.val_size, seed=script_args.seed)
        val_question_tensors = valid_dataset['input_ids']
        val_questions = valid_dataset['query']
    rmformat = anscat
elif "toxicity" in script_args.dataset_name:
    dataset, valid_dataset = build_toxicity_promptdata(tokenizer, num_samples=script_args.val_size, seed=script_args.seed, val_strategy=script_args.val_strategy)
    val_question_tensors = valid_dataset['input_ids']
    val_questions = valid_dataset['query']
    rmformat = anscat
else: 
    pftmp = "default"
    mdatatmp = []
    if "einstein" in script_args.dataset_name: 
        print("einstein data format")
        pftmp = 'einstein'
        mdatatmp = ['sol_rows', 'response_j']
    elif "distil" in script_args.dataset_name or "math" in script_args.dataset_name: 
        pftmp = 'onlyans'
        # mdatatmp = ['response_k', 'response_j']
    # keep track of solution rows
    dataset = build_custom_promptdata(tokenizer, script_args.dataset_name, pftmp, mdatatmp)
if ("math" in script_args.reward_model_name) and ("function" in script_args.reward_model_name): 
    print("beware, using math format")
    rmformat = anscat
print(dataset[0])

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer
)

trainable_params = [
    n for n, p in ppo_trainer.model.named_parameters()
    if p.requires_grad
]

print('--------TRAINABLE PARAMS--------')
print(trainable_params)
print(len(trainable_params))
# print(type(ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.lora_A))
# from peft.tuners.lora import LoraLayer
# print('\nq_proj.Lora_A', isinstance(ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.lora_A, LoraLayer))
# print('\nq_proj', isinstance(ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj, LoraLayer))
# print('\nq_proj info',  ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.r,
#                         ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.lora_alpha,
#                         ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.scaling,
#                         ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.lora_dropout,
#                         )

# import torch
# print('vhead type')
# print(isinstance(ppo_trainer.model.module.v_head.summary, torch.nn.Linear))

# ================================================================
# Load evaluation toxicity model (DIFFERENT from training reward model)
# This enables periodic during-training evaluation like Figure 7 in paper
# ================================================================
eval_classifier = None
test_prompts = None
eval_device = 0

if hasattr(script_args, 'eval_freq') and script_args.eval_freq > 0:
    try:
        # Reuse ppo_trainer's accelerator instead of creating a new one
        eval_device = ppo_trainer.accelerator.local_process_index
        eval_classifier = load_eval_toxicity_model(
            script_args.eval_toxicity_model,
            eval_device,
        )
        # PRE-SAMPLE a FIXED set of test prompts once at startup.
        # The same prompts are reused for every evaluation step, eliminating
        # sampling variance and making eval/toxicity_mean directly comparable
        # across training steps.
        try:
            _all_val = val_question_tensors
            _num_eval = min(script_args.eval_num_samples, len(_all_val))
            _fixed_indices = sorted(random.sample(range(len(_all_val)), _num_eval))
            test_prompts = [_all_val[i] for i in _fixed_indices]
            print(f"  FIXED eval prompts sampled at startup: {len(test_prompts)} / {len(_all_val)}")
            print(f"  (Same prompts will be reused every eval step for stable graphs)")
        except NameError:
            print("  WARNING: val_question_tensors not defined, no test prompts for eval")
            test_prompts = None
        print(f"  Eval frequency: every {script_args.eval_freq} steps")
    except Exception as e:
        import traceback
        print(f"WARNING: Could not load eval toxicity model: {e}")
        traceback.print_exc()
        print("  Continuing without periodic evaluation.")
        eval_classifier = None
        test_prompts = None

# ================================================================
# Monkey-patch ppo_trainer.log_stats to inject periodic evaluation
# This works regardless of which rl_utils.py version is installed.
# ================================================================
if eval_classifier is not None and test_prompts is not None:
    _orig_log_stats = ppo_trainer.log_stats
    _eval_step_counter = [0]  # mutable counter
    _eval_freq = script_args.eval_freq
    _eval_num_samples = script_args.eval_num_samples
    _eval_max_new_tokens = max(script_args.max_length - script_args.min_length, 10)
    _eval_temperature = getattr(script_args, 'temperature', 1.0)

    def _patched_log_stats(stats, batch, rewards, columns_to_log=None):
        # Call original log_stats first
        if columns_to_log is not None:
            _orig_log_stats(stats, batch, rewards, columns_to_log)
        else:
            _orig_log_stats(stats, batch, rewards)
        
        _eval_step_counter[0] += 1
        step = _eval_step_counter[0]
        
        if step % _eval_freq == 0:
            try:
                print(f"\n[Step {step}] Running evaluation on test set...")
                eval_gen_kwargs = {
                    "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
                    "max_new_tokens": _eval_max_new_tokens,
                    "temperature": _eval_temperature,
                }
                eval_metrics = evaluate_toxicity_on_test_set(
                    ppo_trainer=ppo_trainer,
                    tokenizer=tokenizer,
                    test_prompts=test_prompts,
                    eval_classifier=eval_classifier,
                    device=eval_device,
                    num_samples=_eval_num_samples,
                    max_new_tokens=_eval_max_new_tokens,
                    generation_kwargs=eval_gen_kwargs,
                )
                print(f"  Eval toxicity: {eval_metrics['eval/toxicity_mean']:.4f} ± {eval_metrics['eval/toxicity_std']:.4f}")
                print(f"  Toxic fraction (>0.5): {eval_metrics['eval/toxic_frac']:.4f}")
                ppo_trainer.accelerator.log(eval_metrics)
            except Exception as e:
                print(f"  WARNING: Evaluation failed at step {step}: {e}")
                import traceback
                traceback.print_exc()
    
    ppo_trainer.log_stats = _patched_log_stats
    print("  ✓ Periodic eval hooked into ppo_trainer.log_stats")
else:
    if eval_classifier is None and hasattr(script_args, 'eval_freq') and script_args.eval_freq > 0:
        print("  WARNING: eval_classifier is None, skipping periodic eval hook")
    if test_prompts is None and eval_classifier is not None:
        print("  WARNING: test_prompts is None, skipping periodic eval hook")

# ================================================================
# Build eval kwargs dict — pass eval params to training loops if they support them
# This provides a SECOND path for eval (in addition to the monkey-patch above)
# ================================================================
import inspect

_eval_kwargs = {}
if eval_classifier is not None and test_prompts is not None:
    _eval_kwargs = {'eval_classifier': eval_classifier, 'test_prompts': test_prompts}

def _get_eval_kwargs(func):
    """Return eval kwargs only if the function accepts them."""
    try:
        sig = inspect.signature(func)
        if 'eval_classifier' in sig.parameters:
            return _eval_kwargs
    except (ValueError, TypeError):
        pass
    return {}

# TODO customize for different RM code, and different RM input formats
# Run RL pipeline now
if script_args.tracin:
    if script_args.with_validation:
        print("NOTE: TracIn with validation dataset")
        train_loop_with_validation(
            script_args, ppo_trainer, reward_model, tokenizer, rmformat,
            min_length=script_args.min_length,
            val_question_tensors=val_question_tensors,
            val_questions=val_questions,
            reward_tokenizer=reward_tokenizer,
            **_get_eval_kwargs(train_loop_with_validation),
        )
    
    else:
        print("Note: TracIn with valid=train")
        train_loop_one_step(script_args, ppo_trainer, reward_model, tokenizer, rmformat, min_length=script_args.min_length, reward_tokenizer=reward_tokenizer)
        
else:
    print("NOTE: standard training without tracin selection")
    train_loop(
        script_args, ppo_trainer, reward_model, tokenizer, rmformat,
        min_length=script_args.min_length,
        reward_tokenizer=reward_tokenizer,
        **_get_eval_kwargs(train_loop),
    )
# train_loop_one_step(script_args, ppo_trainer, reward_model, tokenizer, rmformat)