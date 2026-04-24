"""Run inference comparing baseline and RAG-augmented prompts."""
from custom_helpers import add_root_to_pythonpath, get_config
add_root_to_pythonpath(n_up=2)

import os
import click

from RAG.task4_vector_storage.task import Searcher
from RAG.task5_inference.task import InferenceClient, compare_generations

CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))


@click.command()
@click.option(
    "--query", 
    default="agony",
    help="Word to define"
)
@click.option(
    "--model-id", 
    default="meta-llama/Llama-3.2-1B-Instruct",
    help="HuggingFace model ID to use for generation"
)
@click.option(
    "--max-tokens", 
    default=60,
    type=int,
    help="Maximum number of tokens to generate"
)
@click.option(
    "--no-retrieval", 
    is_flag=True,
    help="Disable retrieval for RAG"
)
def main(query: str, model_id: str, max_tokens: int, no_retrieval: bool) -> None:
    """Run inference comparing baseline and RAG-augmented prompts."""
    # Check for API token
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN environment variable not set. "
            "Please set this variable with your HuggingFace API token."
        )

    # Initialize searcher if retrieval is enabled
    searcher = None
    if not no_retrieval:
        try:
            click.echo(f"🔄 Loading vector store from {CONFIG['VECTOR_STORE_PATH']}...")
            searcher = Searcher.load(CONFIG['VECTOR_STORE_PATH'])
        except Exception as e:
            click.echo(f"⚠️ Could not load vector store: {e}")
            click.echo("Proceeding without retrieval.")

    # Initialize inference client
    click.echo(f"🤖 Initializing inference client for model: {model_id}")
    client = InferenceClient(token, model_id)

    # Run comparison
    click.echo(f"📝 Generating definitions for: '{query}'")
    compare_generations(query, client, searcher, max_tokens=max_tokens)


if __name__ == "__main__":
    main()