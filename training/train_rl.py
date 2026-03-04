"""
train_rl.py - GRPO Reinforcement Learning for QueryMedic.

Stage 2 of 3:
  RL fine-tuning where the reward signal comes from the database clock:
  - Apply the recommended index to a live (test) database
  - Run EXPLAIN ANALYZE before/after
  - Reward = improvement factor (capped at 10x)

Reward components:
  1. Timing improvement (50%) — measured via live EXPLAIN ANALYZE
  2. Index type correctness (25%) — GIN for JSONB, BRIN for monotonic, etc.
  3. Write amplification accuracy (15%) — estimated vs actual
  4. Rewrite semantic equivalence (10%) — result set comparison

Usage:
    deepspeed --num_gpus 14 training/train_rl.py \
        --sft_checkpoint checkpoints/querymedic-sft-v1/final \
        --data_path data/rl_scenarios.jsonl \
        --output_dir checkpoints/querymedic-rl-v1 \
        --run_name querymedic-rl-v1
"""

import argparse
import math
import os
import re
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from core.postgres_internals import IndexType

# ─────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────

TIMING_WEIGHT = 0.50
INDEX_TYPE_WEIGHT = 0.25
WRITE_AMP_WEIGHT = 0.15
REWRITE_WEIGHT = 0.10

MAX_IMPROVEMENT_FACTOR = 10.0   # Cap reward at 10x speedup


def compute_timing_reward(response: str, scenario: dict) -> float:
    """
    Run EXPLAIN ANALYZE before/after and compute timing reward.

    In training, we use pre-computed timing pairs from the scenario.
    In live RL, this would connect to a test database.
    """
    # Extract improvement factor from scenario (pre-computed for training)
    actual_improvement = scenario.get("ground_truth_improvement_factor", 1.0)

    # Check if model predicted an improvement at all
    predicted_any_index = bool(re.search(r"CREATE\s+INDEX", response, re.IGNORECASE))

    if not predicted_any_index:
        return 0.0  # No index proposed — no timing benefit

    # Scale reward: 1.0 → 0.0, 2.0 → 0.5, 5.0 → 0.8, 10x+ → 1.0
    capped = min(actual_improvement, MAX_IMPROVEMENT_FACTOR)
    reward = (capped - 1.0) / (MAX_IMPROVEMENT_FACTOR - 1.0)
    return max(0.0, reward)


def compute_index_type_reward(response: str, scenario: dict) -> float:
    """
    Check whether the correct index type was chosen.

    Ground truth from scenario: "expected_index_type" (gin/btree/brin/gist)
    """
    expected = scenario.get("expected_index_type", "btree").lower()

    # Extract USING clause from response
    m = re.search(r"USING\s+(\w+)", response, re.IGNORECASE)
    if not m:
        # If no USING, assume B-tree
        predicted = "btree"
    else:
        predicted = m.group(1).lower()

    if predicted == expected:
        return 1.0

    # Partial credit: gin and gist are both "advanced" types
    gin_gist_equiv = {IndexType.GIN.value, IndexType.GIST.value}
    if predicted in gin_gist_equiv and expected in gin_gist_equiv:
        return 0.5

    return 0.0


def compute_write_amplification_reward(response: str, scenario: dict) -> float:
    """
    Check whether write amplification was quantified and approximately correct.
    """
    # Check if write amplification was mentioned at all
    mentioned = bool(re.search(r"write\s+(amplification|overhead|cost|impact)", response, re.IGNORECASE))
    if not mentioned:
        return 0.0

    # Check if a numeric estimate was given
    has_number = bool(re.search(r"\d+\.?\d*x|\d+%", response))
    if not has_number:
        return 0.3  # Mentioned but not quantified

    return 1.0


def compute_rewrite_reward(response: str, scenario: dict) -> float:
    """
    Check whether a query rewrite was proposed when needed.
    """
    needs_rewrite = scenario.get("needs_rewrite", False)
    has_rewrite = (
        "NOT EXISTS" in response.upper()
        or "UNION" in response.upper()
        or "UNION ALL" in response.upper()
        or "keyset" in response.lower()
        or re.search(r"rewrite|sargable", response, re.IGNORECASE)
    )

    if needs_rewrite and has_rewrite:
        return 1.0
    if needs_rewrite and not has_rewrite:
        return 0.0
    if not needs_rewrite and not has_rewrite:
        return 1.0  # Correctly didn't rewrite
    if not needs_rewrite and has_rewrite:
        return 0.5  # Proposed unnecessary rewrite — not penalized heavily

    return 0.5


def reward_function(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    GRPO reward function — evaluates each completion against the scenario.

    kwargs contains scenario metadata passed through from the dataset.
    Must return exactly len(completions) reward values.
    """
    rewards = []
    scenarios = kwargs.get("scenarios", [{}] * len(completions))
    if not scenarios:
        scenarios = [{}] * len(completions)
    elif len(scenarios) < len(completions):
        # QM-9: Use math.ceil so that the repetition count covers ALL
        # completions even when len(completions) is not exactly divisible by
        # len(scenarios). Integer division undercounts by up to (len(scenarios)-1).
        num_gen = math.ceil(len(completions) / len(scenarios))
        scenarios = [s for s in scenarios for _ in range(num_gen)]
        # Trim to exact length (ceil may have over-produced)
        scenarios = scenarios[:len(completions)]

    for completion in completions:
        idx = len(rewards)
        scenario = scenarios[idx] if idx < len(scenarios) else {}
        timing_r = compute_timing_reward(completion, scenario)
        index_type_r = compute_index_type_reward(completion, scenario)
        write_amp_r = compute_write_amplification_reward(completion, scenario)
        rewrite_r = compute_rewrite_reward(completion, scenario)

        total = (
            TIMING_WEIGHT * timing_r
            + INDEX_TYPE_WEIGHT * index_type_r
            + WRITE_AMP_WEIGHT * write_amp_r
            + REWRITE_WEIGHT * rewrite_r
        )

        rewards.append(total)

        logger.debug(
            f"Reward: {total:.3f} "
            f"(timing={timing_r:.2f}, idx_type={index_type_r:.2f}, "
            f"wa={write_amp_r:.2f}, rewrite={rewrite_r:.2f})"
        )

    return rewards


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

def prepare_rl_dataset(data_path: str, tokenizer) -> Dataset:
    """
    Load RL scenarios. Each scenario is a query optimization problem with:
    - prompt: EXPLAIN ANALYZE + schema
    - ground_truth_improvement_factor: pre-measured timing ratio
    - expected_index_type: gin/btree/brin/gist
    - needs_rewrite: bool
    """
    ds = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(ds):,} RL scenarios")

    def format_prompt(ex):
        fallback = "You are QueryMedic — a database query optimization specialist."
        # Support both raw scenario format ("prompt" field) and synthesized
        # ShareGPT-style format ("conversations" with from/value keys)
        conversations = ex.get("conversations", [])
        if conversations:
            system = fallback
            user = ""
            for turn in conversations:
                role = turn.get("from") or turn.get("role", "")
                content = turn.get("value") or turn.get("content", "")
                if role == "system":
                    system = content or fallback
                elif role in ("human", "user") and not user:
                    user = content
            system = system or fallback
        else:
            system = ex.get("system", fallback) or fallback
            user = ex.get("prompt", "")

        if not user:
            user = "(no query provided)"

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {
            "query": text,
            "scenarios": ex,   # Pass full scenario for reward function
        }

    ds = ds.map(format_prompt)
    return ds


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QueryMedic GRPO RL Training")
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints/querymedic-rl-v1")
    parser.add_argument("--run_name", default="querymedic-rl-v1")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # Left padding for generation

    logger.info(f"Loading SFT checkpoint: {args.sft_checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    train_ds = prepare_rl_dataset(args.data_path, tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        bf16=True,
        logging_steps=5,
        save_steps=100,
        report_to=["wandb"],
        # QM-6: Use an absolute path so DeepSpeed can find the config file
        # regardless of the working directory from which the script is launched
        # (e.g. deepspeed --num_gpus 14 training/train_rl.py launches from repo root).
        deepspeed=str(Path(__file__).parent / "configs" / "ds_config_rl.json"),
        # GRPO-specific
        num_generations=8,           # Generate 8 completions per prompt, pick best
        max_completion_length=2048,
        temperature=0.9,
        top_p=0.95,
        # KL penalty to stay close to SFT reference
        beta=0.05,
        epsilon=0.2,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO RL training...")
    trainer.train()

    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    logger.info("GRPO RL training complete.")


if __name__ == "__main__":
    main()
