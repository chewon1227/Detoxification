"""DPO training script for Ko-Llama-3.1 with full/LoRA/QLoRA/GaLore options."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import sys

# Ensure project root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

from src.train.chat_prompt import (
    build_converse_prompt_from_text,
    strip_embedded_prompt,
)
from src.train.train_config import (
    DPOTrainConfig,
    GaLoreConfig,
    LoRAConfig,
    WandBConfig,
    galore_optim_args,
    galore_optim_name,
)


def _dtype_from_str(name: str) -> torch.dtype:
    if name.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name.lower() == "fp16":
        return torch.float16
    return torch.float32


def _load_jsonl(path: Path, max_samples: int | None) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path} line {line_idx + 1} JSON decode error: {e}")
            if max_samples and len(rows) >= max_samples:
                break
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def _build_prompt(tokenizer, system_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _prepare_dataset(tokenizer, config: DPOTrainConfig) -> Dataset:
    raw_rows = _load_jsonl(config.dataset_path, config.max_train_samples)
    rows = []
    for row in raw_rows:
        if config.prompt_format == "instruct":
            prompt_text = _build_prompt(tokenizer, config.system_prompt, row["prompt"])
        else:
            clean_user = strip_embedded_prompt(row.get("prompt", ""))
            if not clean_user:
                continue
            prompt_text = build_converse_prompt_from_text(clean_user, config.system_prompt)
        rows.append(
            {
                "prompt": prompt_text,
                "chosen": (row["chosen"] or "").strip() + tokenizer.eos_token,
                "rejected": (row["rejected"] or "").strip() + tokenizer.eos_token,
            }
        )
    return Dataset.from_list(rows)


def _set_wandb_env(config: DPOTrainConfig) -> None:
    if not config.wandb.enabled:
        return
    try:
        import wandb  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Weights & Biases logging requested but the `wandb` package is not installed. "
            "Install it with `pip install wandb`."
        ) from exc
    os.environ["WANDB_PROJECT"] = config.wandb.project
    if config.wandb.entity:
        os.environ["WANDB_ENTITY"] = config.wandb.entity


def _build_models(config: DPOTrainConfig):
    if config.mode == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.lora.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.lora.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=_dtype_from_str(config.lora.bnb_4bit_compute_dtype),
        )
        base_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
        }
    else:
        base_kwargs = {
            "torch_dtype": _dtype_from_str(config.precision),
            "device_map": "auto",
        }

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **base_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name, **base_kwargs)

    if config.mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    if config.mode in {"lora", "qlora"}:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            task_type="CAUSAL_LM",
            target_modules=config.lora.target_modules,
            use_rslora=config.lora.use_rslora,
        )
        model = get_peft_model(model, lora_config)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    return model, ref_model


def _parse_args() -> DPOTrainConfig:
    parser = argparse.ArgumentParser(description="Run DPO training.")
    defaults = DPOTrainConfig()
    parser.add_argument("--mode", choices=["full", "lora", "qlora", "galore"], default=defaults.mode)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--dataset-path", default=str(defaults.dataset_path))
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--max-prompt-length", type=int, default=defaults.max_prompt_length)
    parser.add_argument("--num-train-epochs", type=float, default=defaults.num_train_epochs)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--logging-steps", type=int, default=defaults.logging_steps)
    parser.add_argument("--logging-strategy", default=defaults.logging_strategy, choices=["steps", "epoch", "no"])
    parser.add_argument("--save-strategy", default=defaults.save_strategy, choices=["steps", "epoch", "no"])
    parser.add_argument("--save-steps", type=int, default=defaults.save_steps)
    parser.add_argument("--eval-steps", type=int, default=defaults.eval_steps)
    parser.add_argument("--eval-strategy", default=defaults.eval_strategy, choices=["steps", "epoch", "no"])
    parser.add_argument("--save-total-limit", type=int, default=defaults.save_total_limit)
    parser.add_argument("--lr-scheduler-type", default=defaults.lr_scheduler_type)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--dataloader-num-workers", type=int, default=defaults.dataloader_num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=defaults.precision)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--system-prompt", default=defaults.system_prompt)
    parser.add_argument("--prompt-format", choices=["instruct", "converse"], default=defaults.prompt_format)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument(
        "--loss-type",
        nargs="+",
        default=defaults.loss_type if isinstance(defaults.loss_type, list) else [defaults.loss_type],
        help="TRL loss type(s); pass one or more values.",
    )
    parser.add_argument("--lora-r", type=int, default=defaults.lora.r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora.alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora.dropout)
    parser.add_argument("--lora-bias", default=defaults.lora.bias)
    parser.add_argument("--lora-target-modules", nargs="+", default=defaults.lora.target_modules)
    parser.add_argument("--use-rslora", action="store_true", default=defaults.lora.use_rslora)
    parser.add_argument("--bnb-4bit-quant-type", default=defaults.lora.bnb_4bit_quant_type)
    parser.add_argument(
        "--bnb-4bit-use-double-quant",
        dest="bnb_4bit_use_double_quant",
        action="store_true",
        default=defaults.lora.bnb_4bit_use_double_quant,
    )
    parser.add_argument(
        "--no-bnb-4bit-use-double-quant",
        dest="bnb_4bit_use_double_quant",
        action="store_false",
        help="Disable double quantization for QLoRA.",
    )
    parser.add_argument("--bnb-4bit-compute-dtype", default=defaults.lora.bnb_4bit_compute_dtype)
    parser.add_argument("--galore-optimizer", choices=["adamw", "adamw_8bit", "adafactor"], default=defaults.galore.optimizer)
    parser.add_argument("--galore-rank", type=int, default=defaults.galore.rank)
    parser.add_argument("--galore-scale", type=float, default=defaults.galore.scale)
    parser.add_argument("--galore-update-proj-gap", type=int, default=defaults.galore.update_proj_gap)
    parser.add_argument("--galore-proj-type", default=defaults.galore.proj_type)
    parser.add_argument(
        "--galore-layerwise",
        action="store_true",
        default=defaults.galore.layerwise,
        help="Enable layer-wise GaLore projections.",
    )
    parser.add_argument("--use-wandb", action="store_true", default=defaults.wandb.enabled, help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=defaults.wandb.project)
    parser.add_argument("--wandb-entity", default=defaults.wandb.entity)
    parser.add_argument("--wandb-run-name", default=defaults.wandb.run_name)
    args = parser.parse_args()

    return DPOTrainConfig(
        mode=args.mode,
        model_name=args.model_name,
        dataset_path=Path(args.dataset_path),
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        logging_strategy=args.logging_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        precision=args.precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        max_train_samples=args.max_train_samples,
        system_prompt=args.system_prompt,
        prompt_format=args.prompt_format,
        beta=args.beta,
        loss_type=args.loss_type[0] if len(args.loss_type) == 1 else args.loss_type,
        lora=LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=args.lora_target_modules,
            use_rslora=args.use_rslora,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        ),
        galore=GaLoreConfig(
            optimizer=args.galore_optimizer,
            rank=args.galore_rank,
            scale=args.galore_scale,
            update_proj_gap=args.galore_update_proj_gap,
            proj_type=args.galore_proj_type,
            layerwise=args.galore_layerwise,
        ),
        wandb=WandBConfig(
            enabled=args.use_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
        ),
    )


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = _parse_args()
    _set_wandb_env(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = _prepare_dataset(tokenizer, config)
    model, ref_model = _build_models(config)

    optim_args = None
    if config.mode in {"lora", "qlora"}:
        optim = "paged_adamw_32bit"
    elif config.mode == "galore":
        optim = galore_optim_name(config.galore)
        optim_args = galore_optim_args(config.galore)
    else:
        optim = "adamw_torch"
    loss_types = (
        [config.loss_type] if isinstance(config.loss_type, str) else list(config.loss_type)
    )
    dpo_args = DPOConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps or 0,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        logging_strategy=config.logging_strategy,
        save_steps=config.save_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=["wandb"] if config.wandb.enabled else ["none"],
        run_name=config.wandb.run_name,
        remove_unused_columns=False,
        optim=optim,
        pad_token=tokenizer.pad_token,
        label_pad_token_id=-100,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_seq_length,
        max_length=config.max_seq_length,
        truncation_mode="keep_start",
        precompute_ref_log_probs=False,
        beta=config.beta,
        loss_type=loss_types,
        optim_args=optim_args,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
