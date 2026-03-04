"""
train.py - Supervised Fine-Tuning (SFT) for QueryMedic.

Stage 1 of 3:
  SFT on (EXPLAIN ANALYZE + schema + query) → (diagnosis + index DDL + rewritten query) pairs.

Hardware: 18x A6000 48GB
Strategy: DeepSpeed ZeRO-3, LoRA r=64, base model Qwen2.5-7B-Coder-Instruct

Usage:
    # 18 GPUs, ZeRO-3:
    deepspeed --num_gpus=18 training/train.py \
        --deepspeed training/configs/deepspeed_zero3.json \
        --data-dir data/training \
        --output-dir checkpoints/querymedic-sft-v1

    # Single node, 8 GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus=8 training/train.py \
        --deepspeed training/configs/deepspeed_zero3.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer

QUERYMEDIC_SYSTEM_PROMPT = """You are QueryMedic — a database query optimization specialist.

Given a slow query, its EXPLAIN ANALYZE output, and schema context, you:
1. Diagnose the root cause (seq scan, stale stats, bad join, missing index, structural issue)
2. Recommend the optimal index (type, column order, partial/covering, write impact)
3. Rewrite the query if structural issues exist (NOT IN→NOT EXISTS, OR→UNION, sargable rewrites)
4. Provide ready-to-execute DDL

Be precise: cite cost numbers, actual row counts, plan nodes.
Quantify write amplification for every index you propose.
Only recommend indexes that will be used — prefer covering indexes over multiple single-column indexes."""

BASE_MODEL = "Qwen/Qwen2.5-7B-Coder-Instruct"
DATA_DIR = Path(__file__).parents[1] / "data" / "training"
OUTPUT_DIR = Path(__file__).parents[1] / "checkpoints" / "querymedic-sft"
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "8192"))


def load_training_data(data_dir: Path) -> Dataset:
    """
    Load SFT training data from data/training directory.

    Supports both a single sharegpt_train.jsonl file and a glob of *.jsonl files.

    Expected JSONL format (ShareGPT):
    {
      "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "## Query\\n...\\n## EXPLAIN ANALYZE\\n..."},
        {"from": "gpt", "value": "## Diagnosis\\n...\\n## Index DDL\\n```sql\\n...```"}
      ]
    }
    """
    records = []

    # First try the standard single-file path
    train_path = data_dir / "sharegpt_train.jsonl"
    if train_path.exists():
        for line in train_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        logger.info(f"Loaded {len(records)} training pairs from {train_path}")
    else:
        # Load all *.jsonl files in the directory
        jsonl_files = sorted(data_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No training data found in {data_dir}. "
                "Run synthesis pipeline first: python synthesis/query_synthesizer.py"
            )
        for fpath in jsonl_files:
            if "val" in fpath.name:
                continue  # Skip validation files
            for line in fpath.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        logger.info(
            f"Loaded {len(records)} training pairs from {len(jsonl_files)} files in {data_dir}"
        )

    return Dataset.from_list(records)


def prepare_dataset(data_dir: Path, tokenizer) -> Dataset:
    """Load and format the SFT dataset for training."""
    ds = load_training_data(data_dir)
    logger.info(f"Loaded {len(ds):,} training examples")

    def format_example(ex):
        convs = ex["conversations"]
        msgs = []
        for turn in convs:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            role = role_map.get(turn["from"], turn["from"])
            msgs.append({"role": role, "content": turn["value"]})
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(format_example, remove_columns=ds.column_names)

    # Filter sequences that are too long
    def filter_length(ex):
        tokens = tokenizer(ex["text"], return_length=True)
        return tokens["length"][0] <= MAX_SEQ_LEN

    before = len(ds)
    ds = ds.filter(filter_length)
    logger.info(
        f"Filtered {before - len(ds)} examples exceeding {MAX_SEQ_LEN} tokens. Remaining: {len(ds):,}"
    )

    return ds


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
    )


def build_training_args(
    output_dir: Path, num_gpus: int, deepspeed_config: str | None
) -> SFTConfig:
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        # QM-4: max(num_gpus, 1) guards against ZeroDivisionError when
        # WORLD_SIZE env var is not set (defaults to 1 above, but be safe).
        gradient_accumulation_steps=max(1, 16 // max(num_gpus, 1)),
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=25,
        # QM-4: evaluation_strategy must be set so eval_steps is honoured.
        # Without it, HuggingFace Trainer defaults to "no" and never evaluates
        # even when a val dataset is provided.
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=3,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name="querymedic-sft",
        deepspeed=deepspeed_config,
        dataloader_num_workers=4,
        group_by_length=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        # SFTConfig-specific: dataset formatting args belong here, not in SFTTrainer
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
    )


class LogMetricsCallback(TrainerCallback):
    """Log training metrics to loguru."""

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        if logs:
            step = state.global_step
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            if isinstance(loss, float):
                logger.info(f"step {step:>6} | loss {loss:.4f} | lr {lr:.2e}")


def main():
    parser = argparse.ArgumentParser(description="QueryMedic SFT Training")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--deepspeed", default=None, help="DeepSpeed config path")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"QueryMedic SFT | GPUs: {num_gpus} | Model: {args.model}")
    logger.info(f"Data: {args.data_dir} | Output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=False,  # disabled during training
    )

    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds = prepare_dataset(args.data_dir, tokenizer)

    # Load optional val set
    val_ds = None
    val_path = args.data_dir / "sharegpt_val.jsonl"
    if val_path.exists():
        val_records = [
            json.loads(line) for line in val_path.read_text().splitlines() if line.strip()
        ]
        if val_records:

            def format_val(ex):
                convs = ex["conversations"]
                msgs = []
                for turn in convs:
                    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
                    role = role_map.get(turn["from"], turn["from"])
                    msgs.append({"role": role, "content": turn["value"]})
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                return {"text": text}

            from datasets import Dataset as HFDataset

            val_ds_raw = HFDataset.from_list(val_records)
            val_ds = val_ds_raw.map(format_val, remove_columns=val_ds_raw.column_names)
            logger.info(f"Loaded {len(val_ds):,} validation examples")

    training_args = build_training_args(args.output_dir, num_gpus, args.deepspeed)
    training_args.num_train_epochs = args.epochs
    training_args.learning_rate = args.lr

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[LogMetricsCallback()],
    )

    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    logger.success(f"SFT complete → {args.output_dir}/final")


if __name__ == "__main__":
    main()
