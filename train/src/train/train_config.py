from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

TrainingMode = Literal["full", "lora", "qlora", "galore"]
PrecisionType = Literal["bf16", "fp16", "fp32"]
GaLoreOptimizer = Literal["adamw", "adamw_8bit", "adafactor"]


def default_lora_targets() -> List[str]:
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


@dataclass
class WandBConfig:
    """Weights & Biases logging options."""

    enabled: bool = False
    project: str = "yaicon"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class GaLoreConfig:
    """GaLore optimizer knobs."""

    optimizer: GaLoreOptimizer = "adamw"
    rank: int = 128
    update_proj_gap: int = 200
    scale: float = 0.25
    proj_type: str = "std"
    layerwise: bool = False


@dataclass
class LoRAConfig:
    """LoRA/QLoRA knobs."""

    r: int = 64 * 2
    alpha: int = 128
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=default_lora_targets)
    use_rslora: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class BaseTrainConfig:
    """Common training knobs used by both SFT and DPO."""

    model_name: str = "s5ya/Ko-Llama-3.1-8B-Lexi-Uncensored-V2"
    mode: TrainingMode = "qlora"
    max_seq_length: int = 2048
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    warmup_steps: Optional[int] = None
    weight_decay: float = 0.01
    logging_steps: int = 10
    logging_strategy: str = "steps"
    save_strategy: str = "steps"
    save_steps: int = 10_000
    eval_steps: int = 500
    eval_strategy: str = "no"
    save_total_limit: int = 2
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4
    seed: int = 42
    gradient_checkpointing: bool = True
    precision: PrecisionType = "bf16"
    max_train_samples: Optional[int] = None
    wandb: WandBConfig = field(default_factory=WandBConfig)


@dataclass
class SFTTrainConfig(BaseTrainConfig):
    dataset_path: Path = Path("dataset/sft_dataset_clean.jsonl")
    system_prompt: str = (
        "커뮤니티 텍스트를 존중하는 말투로 순화하는 조수입니다. "
        "입력은 원문, 출력은 순화된 문장입니다."
    )
    max_prompt_length: int = 1024
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    galore: GaLoreConfig = field(default_factory=GaLoreConfig)
    output_dir: str = field(default='/home/thesol1/yaicon/checkpoints/sft-model')


@dataclass
class DPOTrainConfig(BaseTrainConfig):
    dataset_path: Path = Path("dataset/dpo_dataset.jsonl")
    beta: float = 0.1
    loss_type: Union[str, List[str]] = field(default_factory=lambda: ["sft", "sigmoid"])
    max_prompt_length: int = 1024
    system_prompt: str = (
        "커뮤니티 텍스트를 존중하는 말투로 순화하는 조수입니다. "
        "입력은 원문, 출력은 순화된 문장입니다."
    )
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    galore: GaLoreConfig = field(default_factory=GaLoreConfig)
    output_dir: str = field(default='/home/thesol1/yaicon/checkpoints/dpo-model')


def galore_optim_name(config: GaLoreConfig) -> str:
    base = {
        "adamw": "galore_adamw",
        "adamw_8bit": "galore_adamw_8bit",
        "adafactor": "galore_adafactor",
    }[config.optimizer]
    if config.layerwise:
        base = f"{base}_layerwise"
    return base


def galore_optim_args(config: GaLoreConfig) -> str:
    return (
        f"rank={config.rank},"
        f"update_proj_gap={config.update_proj_gap},"
        f"scale={config.scale},"
        f"proj_type={config.proj_type}"
    )
