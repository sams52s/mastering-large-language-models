from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2)

import os
import json
import logging
import itertools
import pandas as pd
from functools import partial
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union

import torch
import wandb
import transformers
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback
)
from peft import PeftConfig, get_peft_model, PeftModel
from datasets import Dataset

from FineTuning.task5_grid_trainer.callbacks import LoggingCallback, ExampleGenerationCallback
from FineTuning.task4_helpers.task import Helpers
generate_definition_and_example = Helpers.get_example_and_definition
convert_to_serializable = Helpers.convert_to_serializable


class GridTrainer:
    """
    A class to handle training, evaluation, and hyperparameter tuning for PEFT models.

    Attributes:
        training_args (TrainingArguments): Configuration for training.
        base_model (PreTrainedModel): The base model to be fine-tuned.
        base_tokenizer (PreTrainedTokenizer): Tokenizer for the base model.
        train_dataset (Dataset): Dataset for training.
        eval_dataset (Dataset): Dataset for evaluation.
        peft_config (PeftConfig): Configuration for PEFT.
        test_samples (Optional[Dataset]): Optional dataset for generating examples.
    """

    def __init__(
        self,
        training_args: TrainingArguments,
        base_model: PreTrainedModel,
        base_tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        peft_config: PeftConfig,
        test_samples: Optional[Dataset] = None,
    ) -> None:
        self.training_args = training_args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.peft_config = peft_config
        self.test_samples = test_samples

        # Initialize default callbacks
        self.default_callbacks = self._create_default_callbacks()
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

    def _create_default_callbacks(self) -> List[TrainerCallback]:
        """Create default callbacks with initial paths."""
        return [
            ExampleGenerationCallback(
                self.test_samples,
                self.base_tokenizer,
                partial(generate_definition_and_example, params={"max_new_tokens": 20}),
                log_dir=self.training_args.output_dir
            ),
            LoggingCallback(log_path=os.path.join(self.training_args.output_dir, "training_logs.txt"))
        ]

    def _update_callbacks(self, output_dir: str) -> None:
        """Update callback paths with current output directory."""
        for callback in self.default_callbacks:
            if isinstance(callback, ExampleGenerationCallback):
                callback.log_dir = output_dir
            elif isinstance(callback, LoggingCallback):
                callback.log_path = os.path.join(output_dir, "training_logs.txt")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        wandb.init(
            project="peft-training",
            name=self.training_args.run_name,
            config={
                "learning_rate": self.training_args.learning_rate,
                "batch_size": self.training_args.per_device_train_batch_size,
                "epochs": self.training_args.num_train_epochs,
                "peft_config": convert_to_serializable(self.peft_config.to_dict()),
            }
        )

    def _save_file(self, relative_path: str, data: Union[str, bytes]) -> None:
        """Save data to file in current output directory."""
        file_path = Path(self.training_args.output_dir) / relative_path
        with open(file_path, "w" if isinstance(data, str) else "wb") as f:
            f.write(data)

    def train(self, is_grid: bool = False) -> PeftModel:
        """Execute single training run."""
        if not is_grid:
            self._init_wandb()

        # Update callback paths
        self._update_callbacks(self.training_args.output_dir)

        # Setup model and trainer
        peft_model = # TODO: get peft model
        peft_model.print_trainable_parameters()

        trainer = Trainer(
            # TODO: init Trainer. don't forget about data collator, datasets, and callbacks
        )

        # Execute training
        trainer.train()
        
        # Save artifacts
        trainer.save_model(Path(self.training_args.output_dir) / "best_model")
        self._save_file("training_args.json", json.dumps(convert_to_serializable(self.training_args.to_dict()), indent=2))

        if not is_grid:
            wandb.finish()

        return peft_model
    
    def _update_params_grid(self, idx: int, params: Dict[str, Any]) -> None:
        """Update state of training arguments for grid search.
        
        Updates:
        1. Output directory
        2. Run name
        3. Overwrite output directory
        4. WandB config
        5. Callbacks
        6. Training arguments
        """
        if self.training_args.output_dir.endswith(f"run_{idx-1}"):
            self.training_args.output_dir = self.training_args.output_dir[:-len(f"run_{idx-1}")]
        
        # Update output directory
        run_dir = Path(self.training_args.output_dir) / f"run_{idx}"
        self.training_args.output_dir = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Update run name
        if self.training_args.run_name.endswith(f"_run_{idx-1}"):
            self.training_args.run_name = self.training_args.run_name[:-len(f"_run_{idx-1}")]
        
        self.training_args.run_name = f"{self.training_args.run_name}_run_{idx}"
        self.training_args.overwrite_output_dir = True

        # Update training arguments
        self.training_args = TrainingArguments(
            **{**self.training_args.to_dict(), **params}
        )
        
        # Initialize WandB for this run
        self._init_wandb()
        wandb.config.update(params, allow_val_change=True)

        # Save parameters
        self._save_file("hyperparams.json", json.dumps(convert_to_serializable(params), indent=2))
        self._save_file(
            "wandb_config.json",
            json.dumps(convert_to_serializable(wandb.config.as_dict()), indent=2)
        )

        # Update callbacks
        self._update_callbacks(self.training_args.output_dir)

    def _get_params_grid(self, grid_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        pass # TODO: YOUR CODE HERE

    def grid_search(self, grid_params: Dict[str, List[Any]]) -> None:
        """Execute hyperparameter grid search."""
        # Generate parameter combinations
        param_combinations = self._get_params_grid(grid_params)
        if wandb.run is not None:
            wandb.log({"total_runs": len(param_combinations)})

        for i, params in enumerate(param_combinations):
            # TODO: update grid parameters and run training
            wandb.log({"completed_runs": i + 1})

        wandb.finish()
