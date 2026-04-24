"""Bi‑encoder fine‑tuning with LoRA."""
from pathlib import Path
from typing import Union
import os

from datasets import DatasetDict, load_from_disk, disable_caching
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData
)
from sentence_transformers.losses import TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from peft import LoraConfig, TaskType

from custom_helpers import get_config

CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))
disable_caching()


def get_peft_model(
    base_model: str = CONFIG["MODEL_BASE"], 
    r: int = 16, 
    alpha: int = 32
) -> SentenceTransformer:
    """Initialize a PEFT model with LoRA adapters.
    
    Args:
        base_model: Base model identifier
        r: LoRA rank parameter
        alpha: LoRA alpha parameter
        
    Returns:
        SentenceTransformer model with LoRA adapters
    """
    model = SentenceTransformer(
        base_model,
        trust_remote_code=True,
        model_card_data=SentenceTransformerModelCardData(language="en", license="apache-2.0"),
    )
    
    lora_cfg = LoraConfig(
        # TODO: Create LoRA config
    )
    
    # TODO: Add LoRA adapter to the model
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model initialized with {trainable_params:,} trainable parameters")
    if total_params > 0:  # Avoid division by zero (clutch for testing)
        print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} " 
            f"({trainable_params/total_params:.2%})")
    
    return model


def get_datasets(path: Union[str, Path]) -> DatasetDict:
    """Load datasets from disk.
    
    Args:
        path: Path to the datasets directory
        
    Returns:
        DatasetDict with 'train' and 'eval' splits
    """
    dataset = load_from_disk(Path(path))
    print(f"📚 Loaded datasets:")
    print(f"  - Training: {len(dataset['train']):,} examples")
    print(f"  - Evaluation: {len(dataset['eval']):,} examples")
    return dataset


def configure_training(
    model: SentenceTransformer, 
    datasets: DatasetDict, 
    epochs: int = CONFIG["EPOCHS"],
    batch_size: int = CONFIG["BATCH_SIZE"], 
    eval_steps: int = CONFIG["EVAL_STEPS"]
) -> SentenceTransformerTrainer:
    """Configure the sentence transformer training.
    
    Args:
        model: SentenceTransformer model
        datasets: DatasetDict with train and eval splits
        epochs: Number of training epochs
        batch_size: Batch size for training
        eval_steps: Steps between evaluations
        
    Returns:
        Configured SentenceTransformerTrainer
    """
    args = SentenceTransformerTrainingArguments(
        output_dir=CONFIG["MODEL_OUT"],
        load_best_model_at_end=False,  # Don't set this to True; bug in PEFT & SentenceTransformers integration
        # feel free to change the parameters below
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        fp16=True,

        # TODO: add other arguments
    )

    evaluator = # TODO: create evaluator

    trainer = # TODO: create trainer
    
    return trainer


def run_training(trainer: SentenceTransformerTrainer) -> None:
    """Run the training process.
    
    Args:
        trainer: Configured SentenceTransformerTrainer
    """
    print("🚀 Starting training...")
    trainer.train()
    
    # Create output directory if it doesn't exist
    Path(CONFIG["MODEL_OUT"]).mkdir(parents=True, exist_ok=True)
    
    trainer.model.save_pretrained(CONFIG["MODEL_OUT"])
    print(f"✅ Model saved to {CONFIG['MODEL_OUT']}")