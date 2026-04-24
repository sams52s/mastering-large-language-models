"""Run bi-encoder fine-tuning with LoRA."""
from custom_helpers import add_root_to_pythonpath, get_config
add_root_to_pythonpath(n_up=2)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import click
from pathlib import Path

from RAG.task3_training.task import (
    get_peft_model, 
    get_datasets, 
    configure_training, 
    run_training
)

CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))


@click.command()
@click.option(
    "--data", 
    default=CONFIG["RAG_DATASET_DIR"],
    help="Path to the processed dataset directory"
)
@click.option(
    "--epochs", 
    default=CONFIG["EPOCHS"],
    help="Number of training epochs"
)
def main(data: str, epochs: int) -> None:
    """Run bi-encoder fine-tuning with LoRA."""
    data_path = Path(data)
    if not data_path.exists():
        click.echo(f"❌ Dataset path does not exist: {data_path}")
        return
    
    click.echo(f"🔄 Initializing PEFT model...")
    model = get_peft_model()
    
    click.echo(f"📚 Loading datasets from {data_path}...")
    datasets = get_datasets(data_path)
    
    click.echo(f"🛠️ Configuring training (epochs={epochs})...")
    trainer = configure_training(model, datasets, epochs=epochs)
    
    click.echo(f"🚀 Starting training...")
    run_training(trainer)


if __name__ == "__main__":
    main()