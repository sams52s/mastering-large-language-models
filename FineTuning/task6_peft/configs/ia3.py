from custom_helpers import get_config
from .base import BaseConfig
from peft import (
    IA3Config,
    TaskType,
    PeftConfig
)
from typing import Dict


class PeftIA3Config(BaseConfig):
    """Configuration class for IA3 PEFT model training."""
    
    @property
    def model_config(self) -> Dict:
        return {
            "model_name": self.conf["model"],
            "quantize": False,
            "is_prompt_tuning": False
        }
    
    @property
    def peft_config(self) -> PeftConfig:
        return IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
            inference_mode=False
        )

    @property
    def training_config(self) -> Dict:
        return {
            **self.DEFAULT_TRAINING_CONFIG,
            "output_dir": self.conf.exp.ia3_dir
        }
