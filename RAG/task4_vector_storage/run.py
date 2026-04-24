"""Run vector search on definitions."""
from custom_helpers import add_root_to_pythonpath, get_config
add_root_to_pythonpath(n_up=2)

import click
from pathlib import Path
import os

import RAG.task3_training.run as task3_run  # just to set CUDA_VISIBLE_DEVICES
from RAG.task4_vector_storage.task import load_model, load_definitions, Searcher

CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))


@click.command()
@click.option(
    "--model", 
    default=CONFIG["MODEL_OUT"],
    help="Path to the sentence transformer model"
)
@click.option(
    "--definitions", 
    default=CONFIG["DEFINITIONS_PATH"],
    help="Path to the definitions JSON file"
)
@click.option(
    "--query", 
    default="agony",
    help="Query word to search for"
)
@click.option(
    "--top-k", 
    default=5,
    type=int,
    help="Number of results to return"
)
@click.option(
    "--save", 
    is_flag=True,
    help="Save the vector store to disk"
)
@click.option(
    "--reuse", 
    is_flag=True,
    help="Reuse cached vector store if present"
)
def main(model: str, definitions: str, query: str, top_k: int, save: bool, reuse: bool) -> None:
    """Run vector search on definitions."""
    vector_store_path = CONFIG["VECTOR_STORE_PATH"]
    
    # Try to reuse cached vector store if requested
    if reuse and Path(vector_store_path).exists():
        click.echo(f"🔄 Loading cached vector store...")
        try:
            searcher = Searcher.load(vector_store_path)
        except Exception as e:
            click.echo(f"⚠️ Failed to load cached vector store: {e}")
            reuse = False
    
    # Create new vector store if not reusing
    if not reuse:
        click.echo(f"🔄 Loading model from {model}...")
        model_instance = load_model(model)
        
        click.echo(f"📚 Loading definitions from {definitions}...")
        definitions_data = load_definitions(definitions)
        
        click.echo(f"🔍 Creating vector store...")
        searcher = Searcher(definitions_data, model_instance)
        
        # Save vector store if requested
        if save:
            click.echo(f"💾 Saving vector store...")
            searcher.save(vector_store_path)

    # Run search
    results = searcher.search(query, k=top_k)
    
    # Display results
    click.echo(f"\n🔎 Top‑{top_k} results for \"{query}\":")
    for rank, (text, score) in enumerate(results, 1):
        click.echo(f"{rank:>2}. {score:0.3f}  {text}")


if __name__ == "__main__":
    main()