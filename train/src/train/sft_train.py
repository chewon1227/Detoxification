"""Supervised fine-tuning script for Ko-Llama-3.1 with full/LoRA/QLoRA/GaLore options."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.train.chat_prompt import (
    build_converse_prompt,
    build_dc_comment_prompt,
    default_personas,
    strip_embedded_prompt,
)
from src.train.train_config import (
    GaLoreConfig,
    LoRAConfig,
    SFTTrainConfig,
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


def _build_converse_prompt_from_context(
    context_text: str, init_persona: Dict[str, str], target_persona: Dict[str, str]
) -> str | None:
    cleaned = strip_embedded_prompt(context_text or "")
    if not cleaned:
        return None
    context = f"{target_persona['name']}: {cleaned.strip()}"
    return build_converse_prompt(init_persona, target_persona, context)


def _build_dc_prompt_from_context(context_text: str, tokenizer) -> str | None:
    cleaned = strip_embedded_prompt(context_text or "")
    if not cleaned:
        return None
    return build_dc_comment_prompt(tokenizer, cleaned.strip())


def _rows_from_messages(
    messages: List[Dict], init_persona: Dict[str, str], target_persona: Dict[str, str], eos_token: str
) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    pending_user: str | None = None
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            pending_user = strip_embedded_prompt(content)
        elif role == "assistant":
            if not pending_user:
                continue
            prompt = _build_converse_prompt_from_context(
                pending_user, init_persona, target_persona
            )
            pending_user = None
            if not prompt:
                continue
            answer = content.strip()
            if not answer:
                continue
            prepared.append(
                {
                    "prompt": prompt,
                    "answer": answer + eos_token,
                }
            )
    return prepared


def _rows_from_messages_dc(
    messages: List[Dict], tokenizer, eos_token: str
) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    pending_user: str | None = None
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            pending_user = strip_embedded_prompt(content)
        elif role == "assistant":
            if not pending_user:
                continue
            prompt = _build_dc_prompt_from_context(pending_user, tokenizer)
            pending_user = None
            if not prompt:
                continue
            answer = content.strip()
            if not answer:
                continue
            prepared.append(
                {
                    "prompt": prompt,
                    "answer": answer + eos_token,
                }
            )
    return prepared


def _build_converse_rows(raw_rows: List[Dict], tokenizer, config: SFTTrainConfig) -> List[Dict[str, str]]:
    init_persona, target_persona = default_personas(config.system_prompt)
    prepared: List[Dict[str, str]] = []
    for row in raw_rows:
        if "messages" in row:
            prepared.extend(
                _rows_from_messages(
                    row["messages"], init_persona, target_persona, tokenizer.eos_token
                )
            )
        else:
            prompt = _build_converse_prompt_from_context(
                row.get("original_text", ""), init_persona, target_persona
            )
            answer = (row.get("detox_text") or "").strip()
            if not prompt or not answer:
                continue
            prepared.append({"prompt": prompt, "answer": answer + tokenizer.eos_token})
    if not prepared:
        raise SystemExit("No usable rows found for converse-style prompts.")
    return prepared


def _build_dc_rows(raw_rows: List[Dict], tokenizer, config: SFTTrainConfig) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    for row in raw_rows:
        if "messages" in row:
            prepared.extend(
                _rows_from_messages_dc(row["messages"], tokenizer, tokenizer.eos_token)
            )
        else:
            prompt = _build_dc_prompt_from_context(row.get("original_text", ""), tokenizer)
            answer = (row.get("detox_text") or "").strip()
            if not prompt or not answer:
                continue
            prepared.append({"prompt": prompt, "answer": answer + tokenizer.eos_token})
    if not prepared:
        raise SystemExit("No usable rows found for dc_comment prompts.")
    return prepared


def _tokenize_prompt_and_answer(row: Dict, tokenizer, config: SFTTrainConfig) -> Dict:
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]

    if len(prompt_ids) > config.max_prompt_length:
        prompt_ids = prompt_ids[-config.max_prompt_length :]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > config.max_seq_length:
        overflow = len(input_ids) - config.max_seq_length
        if overflow < len(prompt_ids):
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
        else:
            input_ids = input_ids[-config.max_seq_length :]
            labels = labels[-config.max_seq_length :]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _prepare_dataset(
    tokenizer,
    config: SFTTrainConfig,
    dataset_path: Path | None = None,
    max_samples: int | None = None,
) -> Dataset:
    target_path = dataset_path or config.dataset_path
    raw_rows = _load_jsonl(target_path, max_samples)
    if config.prompt_format == "instruct":
        prepared_rows = [
            {
                "prompt": _build_prompt(tokenizer, config.system_prompt, row["original_text"]),
                "answer": ((row.get("detox_text") or "").strip() + tokenizer.eos_token),
            }
            for row in raw_rows
        ]
    elif config.prompt_format == "converse":
        prepared_rows = _build_converse_rows(raw_rows, tokenizer, config)
    else:
        prepared_rows = _build_dc_rows(raw_rows, tokenizer, config)

    dataset = Dataset.from_list(prepared_rows)
    return dataset.map(
        lambda row: _tokenize_prompt_and_answer(row, tokenizer, config),
        remove_columns=dataset.column_names,
    )


def _build_eval_dataset(tokenizer, config: SFTTrainConfig, train_dataset: Dataset) -> Optional[Dataset]:
    if config.eval_dataset_path:
        return _prepare_dataset(
            tokenizer,
            config,
            dataset_path=config.eval_dataset_path,
            max_samples=config.max_eval_samples,
        )
    if config.max_eval_samples:
        take = min(config.max_eval_samples, len(train_dataset))
        if take > 0:
            return train_dataset.select(list(range(take)))
    return None


def _build_lm_metrics_fn():
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds_tensor = torch.tensor(predictions, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        valid_mask = labels_tensor != -100
        total_tokens = valid_mask.sum().item()
        if total_tokens == 0:
            return {
                "log_likelihood": float("nan"),
                "cross_entropy": float("nan"),
                "perplexity": float("nan"),
                "token_accuracy": float("nan"),
            }
        safe_labels = labels_tensor.masked_fill(~valid_mask, 0)
        log_probs = preds_tensor.log_softmax(dim=-1)
        label_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        label_log_probs = label_log_probs.masked_fill(~valid_mask, 0.0)
        sum_log_probs = label_log_probs.sum().item()
        avg_log_likelihood = sum_log_probs / total_tokens
        cross_entropy = -avg_log_likelihood
        perplexity = math.exp(min(cross_entropy, 100.0))
        predictions_ids = preds_tensor.argmax(dim=-1)
        correct = ((predictions_ids == labels_tensor) & valid_mask).sum().item()
        token_accuracy = correct / total_tokens
        return {
            "log_likelihood": avg_log_likelihood,
            "cross_entropy": cross_entropy,
            "perplexity": perplexity,
            "token_accuracy": token_accuracy,
        }

    return compute_metrics


def _set_wandb_env(config: SFTTrainConfig) -> None:
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


def _build_model(tokenizer, config: SFTTrainConfig):
    if config.mode == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.lora.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.lora.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=_dtype_from_str(config.lora.bnb_4bit_compute_dtype),
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=_dtype_from_str(config.precision),
            device_map="auto",
        )

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

    return model


def _parse_args() -> SFTTrainConfig:
    parser = argparse.ArgumentParser(description="Run SFT training.")
    defaults = SFTTrainConfig()
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
    parser.add_argument(
        "--eval-dataset-path",
        default=str(defaults.eval_dataset_path) if defaults.eval_dataset_path else None,
        help="Optional JSONL file for evaluation. Defaults to the training dataset when omitted.",
    )
    parser.add_argument("--max-eval-samples", type=int, default=defaults.max_eval_samples)
    parser.add_argument("--prompt-format", choices=["instruct", "converse", "dc_comment"], default=defaults.prompt_format)
    parser.add_argument("--system-prompt", default=defaults.system_prompt)
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

    return SFTTrainConfig(
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
        eval_dataset_path=Path(args.eval_dataset_path) if args.eval_dataset_path else None,
        max_eval_samples=args.max_eval_samples,
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
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
    config = _parse_args()
    _set_wandb_env(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = _prepare_dataset(
        tokenizer,
        config,
        dataset_path=config.dataset_path,
        max_samples=config.max_train_samples,
    )
    eval_dataset = _build_eval_dataset(tokenizer, config, train_dataset)
    model = _build_model(tokenizer, config)

    optim_args = None
    if config.mode in {"lora", "qlora"}:
        optim = "paged_adamw_32bit"
    elif config.mode == "galore":
        optim = galore_optim_name(config.galore)
        optim_args = galore_optim_args(config.galore)
    else:
        optim = "adamw_torch"

    def _supports_arg(name: str) -> bool:
        return name in TrainingArguments.__dataclass_fields__

    ta_kwargs = dict(
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
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="wandb" if config.wandb.enabled else "none",
        optim=optim,
    )
    if optim_args:
        ta_kwargs["optim_args"] = optim_args
    if config.wandb.run_name:
        ta_kwargs["run_name"] = config.wandb.run_name
    if _supports_arg("logging_strategy"):
        ta_kwargs["logging_strategy"] = config.logging_strategy
    if _supports_arg("save_strategy"):
        ta_kwargs["save_strategy"] = config.save_strategy
    if _supports_arg("evaluation_strategy"):
        ta_kwargs["evaluation_strategy"] = config.eval_strategy
        ta_kwargs["eval_steps"] = config.eval_steps
    else:
        ta_kwargs["do_eval"] = config.eval_strategy != "no"
        if config.eval_steps:
            ta_kwargs["eval_steps"] = config.eval_steps

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
        compute_metrics=_build_lm_metrics_fn() if eval_dataset else None,
    )
    trainer.train()
    if eval_dataset is not None and config.eval_strategy == "no":
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    trainer.save_model()


if __name__ == "__main__":
    main()
