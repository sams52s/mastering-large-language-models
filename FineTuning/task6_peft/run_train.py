"""
run.py - PEFT Training Pipeline for Language Model Fine-Tuning
"""
from custom_helpers import add_root_to_pythonpath, get_config
add_root_to_pythonpath(n_up=2)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
import warnings
from typing import Dict, List, Tuple

import torch
import click
from transformers import TrainingArguments
from peft import PeftConfig

from FineTuning.task3_dataset.task import DatasetHandler
from FineTuning.task4_helpers.task import Helpers
from FineTuning.task5_grid_trainer.task import GridTrainer
from FineTuning.task5_grid_trainer.best_run_searcher import BestRunSearcher
from configs import PeftIA3Config, PeftLoRAConfig, PeftPromptTuningConfig, BaseConfig

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)






def load_components(config: BaseConfig) -> Tuple:
    """
    Load model, tokenizer, and datasets
    Returns: (base_model, base_tokenizer, train_dataset, test_dataset, sample_words)
    """
    # Load model and tokenizer
    base_model, base_tokenizer = Helpers.load_model_and_tokenizer(**config.model_config)
    
    # Load datasets
    train, test = DatasetHandler.load(load_dir=config.conf["data"]["train_test_dir"])
    
    # Load sample words
    N_SAMPLE = 5
    with open(config.conf["data"]["words"], "r") as f:
        words = json.load(f)
    sample = random.sample(words, N_SAMPLE)
    
    return base_model, base_tokenizer, train, test, sample

def choose_config(method: str, conf_path: str) -> BaseConfig:
    """
    Return the configuration instance based on the provided method.
    """
    if method == "prompt_tuning":
        return PeftPromptTuningConfig(conf_path)
    elif method == "ia3":
        return PeftIA3Config(conf_path)
    elif method == "lora":
        return PeftLoRAConfig(conf_path)
    else:
        raise ValueError(f"Invalid method: {method}")

def get_training_args(config: BaseConfig) -> TrainingArguments:
    """Return TrainingArguments instance"""
    return TrainingArguments(
        output_dir=config.training_config["output_dir"],
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=config.training_config["eval_steps"],
        save_strategy="steps",
        save_steps=config.training_config["eval_steps"],
        logging_dir="./logs",
        logging_steps=config.training_config["eval_steps"],
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=config.training_config["learning_rates"][3],  # Default value
        num_train_epochs=config.training_config["num_epochs"],
        per_device_train_batch_size=config.training_config["batch_size"],
        per_device_eval_batch_size=config.training_config["batch_size"],
        gradient_accumulation_steps=4,
        fp16=True,
        dataloader_num_workers=4,
        seed=42
    )


def prepare_trainer(
    base_model,
    base_tokenizer,
    train_dataset,
    eval_dataset,
    sample_words,
    peft_config: PeftConfig,
    training_args: TrainingArguments
) -> GridTrainer:
    """Initialize and return GridTrainer instance"""
    return GridTrainer(
        training_args=training_args,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        test_samples=sample_words
    )


@click.command()
@click.option("--method", type=click.Choice(["lora", "prompt_tuning", "ia3"]), required=True)
def main(method: str):
    # Load configuration
    conf_path = "../conf.yaml"
    config = choose_config(method, conf_path)
    
    # Load core components
    base_model, base_tokenizer, train, test, sample = load_components(config)
    base_model.use_position_ids = False  # Disable position ids (to avoid warning since they are not trained in PEFT)
    
    # Prepare training arguments
    training_args = get_training_args(config)
    
    # Initialize trainer
    trainer = prepare_trainer(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        train_dataset=train,
        eval_dataset=test,
        sample_words=sample,
        peft_config=config.peft_config,
        training_args=training_args
    )
    
    # Perform grid search
    trainer.grid_search({"learning_rate": config.training_config["learning_rates"]})
    

if __name__ == "__main__":
    main()