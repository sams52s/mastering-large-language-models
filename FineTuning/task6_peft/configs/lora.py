from custom_helpers import get_config
from .base import BaseConfig
from peft import (
    LoraConfig,
    TaskType,
    PeftConfig
)
from typing import Dict


class PeftLoRAConfig(BaseConfig):
    """Configuration class for LoRA PEFT model training."""
    
    @property
    def model_config(self) -> Dict:
        return {
            "model_name": self.conf["model"],
            "quantize": False,
            "is_prompt_tuning": False
        }
    
    @property
    def peft_config(self) -> PeftConfig:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            inference_mode=False        )
            


    @property
    def training_config(self) -> Dict:
        return {
            **self.DEFAULT_TRAINING_CONFIG,
            "output_dir": self.conf.exp.lora_dir
        }
