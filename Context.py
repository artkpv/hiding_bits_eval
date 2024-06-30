from peft import LoraConfig

from trl import PPOConfig
from simple_parsing import Serializable, field

from utils import SEED

from dataclasses import dataclass
from typing import List


@dataclass
class Context(Serializable):

    bits: List[str] = field(
        default_factory=lambda: [
            "red",
            "blue",
            "yellow",
            "green",
            "cyan",
            "magenta",
            "white",
            "black",
        ]
    )

    ds_size: int = 1000
    """ Size of dataset to use for training."""

    train_test_split: float = 0.2

    generation_kwargs: dict = field(
        default_factory=lambda: dict(
            do_sample=True,
            temperature=0.7,
            top_k=0.0,
            top_p=1.0,
            min_length=-1,
            max_new_tokens=300,
        )
    )


@dataclass
class EvalContext(Context):
    models: List[str] = field(
        default_factory=lambda: [
            "aichat:openai:gpt-4-turbo-preview",
            "aichat:gemini:gemini-pro",
        ]
    )

    overseer: str = "aichat:openai:gpt-3.5-turbo-1106"


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
@dataclass
class TrainContext(Context):
    model_name: str = MODEL_NAME

    load_in_8bit: bool = True
    load_in_4bit: bool = False

    epoch_num: int = 200

    lora_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )

    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            adap_kl_ctrl=False,
            batch_size=64,
            cliprange=0.2,
            cliprange_value=0.2,
            early_stopping=True,
            gamma=1,
            gradient_accumulation_steps=1,
            horizon=10000,
            init_kl_coef=0.2,
            lam=0.95,
            learning_rate=1.4e-5,
            #log_with="wandb",
            mini_batch_size=1,
            model_name=MODEL_NAME,
            optimize_cuda_cache=True,
            ppo_epochs=4,
            seed=SEED,
            target_kl=0.1,
            vf_coef=1,
            remove_unused_columns=False,
        )
    )

    # adafactor_config = dict(
    #    lr=1e-6,
    #    # eps=(1e-30, 1e-3),
    #    clip_threshold=1.0,
    #    decay_rate=-0.8,
    #    beta1=0.9,
    #    weight_decay=0.1,
    #    scale_parameter=False,
    #    relative_step=False,
    #    warmup_init=False,
    # )
