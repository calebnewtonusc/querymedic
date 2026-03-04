"""
train_dpo.py - Direct Preference Optimization (DPO) for QueryMedic.

Stage 3 of 3:
  DPO on human-curated preference pairs where:
  - chosen: response that correctly identifies index type AND quantifies write amplification
  - rejected: response that recommends wrong index type OR ignores write cost

DPO teaches the model to prefer:
  1. Correct index type selection (GIN not B-tree for JSONB)
  2. Write amplification quantification (not just "adds overhead")
  3. Covering index recommendations (not just indexed column)
  4. Partial index recommendations (when WHERE filters >50% rows)

Usage:
    deepspeed --num_gpus 14 training/train_dpo.py \
        --rl_checkpoint checkpoints/querymedic-rl-v1/final \
        --data_path data/dpo_pairs.jsonl \
        --output_dir checkpoints/querymedic-dpo-v1 \
        --run_name querymedic-dpo-v1
"""

import argparse
import os

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig

BASE_MODEL_DEFAULT = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Coder-Instruct")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "8192"))


def prepare_dpo_dataset(data_path: str, tokenizer) -> Dataset:
    """
    Load DPO preference pairs.

    Expected JSONL format:
    {
      "prompt": "## Query\\n...\\n## EXPLAIN ANALYZE\\n...",
      "chosen": "## Diagnosis\\n...\\n## DDL\\n```sql\\nCREATE INDEX ... USING gin ...```\\n## Write Amplification\\n2.5x writes...",
      "rejected": "## Diagnosis\\n...\\n## DDL\\n```sql\\nCREATE INDEX ... USING btree ...```\\nThis will be faster."
    }
    """
    logger.info(f"Loading DPO dataset from {data_path}")
    ds = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(ds):,} preference pairs")

    def format_pair(ex):
        system = ex.get("system", "You are QueryMedic — a database query optimization specialist.")

        def make_messages(response: str) -> list[dict]:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": response},
            ]

        chosen_text = tokenizer.apply_chat_template(
            make_messages(ex["chosen"]), tokenize=False, add_generation_prompt=False
        )
        rejected_text = tokenizer.apply_chat_template(
            make_messages(ex["rejected"]), tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            make_messages("")[:-1],  # Remove empty assistant turn
            tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    ds = ds.map(format_pair, remove_columns=ds.column_names)

    # Filter length
    def filter_length(ex):
        chosen_len = tokenizer(ex["chosen"], return_length=True)["length"][0]
        rejected_len = tokenizer(ex["rejected"], return_length=True)["length"][0]
        return max(chosen_len, rejected_len) <= MAX_SEQ_LEN

    before = len(ds)
    ds = ds.filter(filter_length)
    logger.info(f"Filtered {before - len(ds)} pairs exceeding max length. Remaining: {len(ds):,}")

    return ds


def main():
    parser = argparse.ArgumentParser(description="QueryMedic DPO Training")
    parser.add_argument("--rl_checkpoint", required=True, help="Path to GRPO RL checkpoint")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints/querymedic-dpo-v1")
    parser.add_argument("--run_name", default="querymedic-dpo-v1")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.rl_checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # QM-2: DPO requires right-padding so that label positions align with
    # the assistant response tokens. Left-padding shifts all token positions
    # and causes the loss to be computed on the wrong tokens.
    tokenizer.padding_side = "right"

    logger.info(f"Loading RL checkpoint: {args.rl_checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.rl_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Frozen reference model — stays at RL checkpoint weights
    logger.info("Loading frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.rl_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    train_ds = prepare_dpo_dataset(args.data_path, tokenizer)

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        bf16=True,
        logging_steps=5,
        save_steps=100,
        report_to=["wandb"],
        deepspeed="training/configs/ds_config.json",
        # DPO-specific
        beta=0.1,                       # KL penalty strength (lower = more divergence allowed)
        loss_type="sigmoid",            # Standard DPO loss
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_SEQ_LEN // 2,
        label_smoothing=0.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    logger.info("DPO training complete. Final model saved.")


if __name__ == "__main__":
    main()
