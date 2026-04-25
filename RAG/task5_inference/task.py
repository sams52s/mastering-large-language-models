"""Inference utilities contrasting baseline vs RAG‑augmented prompts."""
from custom_helpers import add_root_to_pythonpath, get_config
add_root_to_pythonpath(n_up=2)

import os
import textwrap
import requests
from typing import List, Literal, Optional
from huggingface_hub import InferenceClient as HFInferenceClient

from RAG.task4_vector_storage.task import Searcher


class InferenceClient:
    """Thin wrapper around the HF text-generation endpoint with updated API."""
    
    def __init__(self, api_token: str, model_id: str = "meta-llama/Llama-3.2-1B-Instruct", provider: str = "novita"):
        """Initialize client.
        
        Args:
            api_token: HuggingFace API token
            model_id: Model ID to use for generation (default: meta-llama/Llama-3.2-1B-Instruct)
            provider: Inference provider to use (default: novita)
        """
        self.model_id = model_id
        self.hf_client = HFInferenceClient(
            provider=provider,
            api_key=api_token,
        )
        print(f"🤖 Initialized inference client for provider: {provider}, default model: {model_id}")

    def generate(self, prompt: str, max_tokens: int = 60, model_id: str = None) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            model_id: Optional override for the model ID (defaults to the one set during initialization)
            
        Returns:
            Generated text as string
        """
        # Use the model_id from the parameters if provided, otherwise use the default one
        model = model_id or self.model_id
        
        # Create a messages array as expected by the new API
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Make the API call
        completion = self.hf_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        
        # Extract just the content string from the response
        return completion.choices[0].message.content


# Default prompt header for dictionary assistant
PROMPT_HEADER = "You are an advanced dictionary assistant. Given a word, provide a clear, one‑sentence definition."


def build_prompt(query: str, definitions: List[str], mode: Literal["baseline", "rag"]) -> str:
    """Build prompt for generation.
    
    Args:
        query: Query word
        definitions: List of retrieved definitions (only used in RAG mode)
        mode: Either "baseline" or "rag"
        
    Returns:
        Formatted prompt string
    """
    if mode == "baseline":
        return f"{PROMPT_HEADER}\n\nWord: {query}\nDefinition:"
    else:  # RAG-enhanced
        context = "\n".join(f"- {definition}" for definition in definitions)
        return (
            f"{PROMPT_HEADER}\n\n"
            f"Use the following retrieved definitions as context:\n"
            f"{context}\n\n"
            f"Word: {query}\nDefinition:"
        )


def compare_generations(
    query: str, 
    client: InferenceClient, 
    searcher: Optional[Searcher] = None, 
    max_tokens: int = 60
) -> None:
    """Compare baseline and RAG-augmented generations.
    
    Args:
        query: Query word
        client: InferenceClient for text generation
        searcher: Optional Searcher for retrieving definitions
        max_tokens: Maximum tokens to generate
    """
    # Get retrieved definitions if a searcher is provided
    retrieved = []
    if searcher:
        print(f"🔍 Retrieving definitions for '{query}'...")
        retrieved = [d for d, _ in searcher.search(query, k=3)]
        print(f"📚 Retrieved {len(retrieved)} definitions")
    
    # Build prompts
    p_base = build_prompt(query, [], mode="baseline")
    p_rag = build_prompt(query, retrieved, mode="rag")
    
    # Generate baseline response
    print("\n── Baseline (no retrieval) ──")
    baseline_response = client.generate(p_base, max_tokens=max_tokens)
    print(baseline_response)
    
    # Generate RAG-augmented response
    print("\n── RAG‑augmented ──")
    rag_response = client.generate(p_rag, max_tokens=max_tokens)
    print(rag_response)