"""Simple vector search with dot‑product (cosine) similarity."""
import json
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Union, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from custom_helpers import get_config

CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))


class SimpleVectorStore:
    """Simple vector store for similarity search with cosine similarity."""
    
    def __init__(self, embeddings: np.ndarray, model: SentenceTransformer):
        """Initialize vector store.
        
        Args:
            embeddings: Array of embedding vectors
            model: SentenceTransformer model for encoding queries
            
        Raises:
            ValueError: If embeddings is not a 2D array
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2‑D")
        self.embeddings = embeddings.astype(np.float32)
        self.model = model
        self.dim = self.embeddings.shape[1]

        self._normalize_embeddings()

    def _normalize_embeddings(self) -> None:
        """Normalize the embeddings in place to unit length."""
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

    def _encode_and_normalize(self, text: str) -> np.ndarray:
        """Encode text and normalize the resulting vector.

        Args:
            text: Input text to encode

        Returns:
            Normalized embedding vector
        """
        embedding = self.model.encode(text)
        embedding = np.asarray(embedding, dtype=np.float32).squeeze()

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding

        return embedding / norm
    def search(self, query: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Tuple of (indices, scores) for the top-k results
        """
        k = min(k, self.embeddings.shape[0])
        q = self._encode_and_normalize(query)
        scores = self.embeddings @ q  # cosine similarity because vectors are unit length

        idx = np.argsort(scores)[::-1][:k]

        return idx, scores[idx]


class Searcher:
    """High‑level helper that manages a SimpleVectorStore."""

    def __init__(self, data: List[str], model: SentenceTransformer):
        """Initialize searcher.
        
        Args:
            data: List of text items for search
            model: SentenceTransformer model for encoding
        """
        print(f"📊 Creating embeddings for {len(data)} items...")
        emb = model.encode(data, convert_to_numpy=True, show_progress_bar=True)

        self.data = data
        self.store = SimpleVectorStore(emb, model)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar items.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (text, score) tuples for the top-k results
        """
        idx, scores = self.store.search(query, k)
        return [(self.data[i], float(score)) for i, score in zip(idx, scores)]

    def save(self, path: Union[str, Path, None] = None) -> None:
        """Save searcher to disk.
        
        Args:
            path: Path to save to, defaults to CONFIG["VECTOR_STORE_PATH"]
        """
        path = Path(path or CONFIG["VECTOR_STORE_PATH"])
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("wb") as f:
            pickle.dump(self, f)
        print(f"💾 Searcher saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> Any:
        """Load searcher from disk.
        
        Args:
            path: Path to load from, defaults to CONFIG["VECTOR_STORE_PATH"]
            
        Returns:
            Loaded Searcher instance
        """
        path = Path(path or CONFIG["VECTOR_STORE_PATH"])
        with path.open("rb") as f:
            searcher = pickle.load(f)
        print(f"📂 Loaded searcher from {path} with {len(searcher.data)} items")
        return searcher


def load_definitions(path: Union[str, Path]) -> List[str]:
    """Load definitions from a JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        List of definition strings
    """
    with Path(path).open() as f:
        definitions = json.load(f)
    print(f"📚 Loaded {len(definitions)} definitions from {path}")
    return definitions


def load_model(path: Union[str, Path]) -> SentenceTransformer:
    """Load a SentenceTransformer model.
    
    Args:
        path: Path to model directory
        
    Returns:
        Loaded SentenceTransformer model
    """
    model = SentenceTransformer(str(path), trust_remote_code=True)
    print(f"🧠 Loaded model from {path}")
    return model