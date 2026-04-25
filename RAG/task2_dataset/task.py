"""Triplet dataset builder for RAG bi‑encoder training."""
import json
from pathlib import Path
from typing import List, Dict, Union
import os

from datasets import Dataset, DatasetDict, concatenate_datasets
from custom_helpers import get_config

CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))


class TripletDatasetBuilder:
    """Utilities to create `(anchor, positive, negative)` triplets."""

    @staticmethod
    def load_raw(path: Union[str, Path]) -> List[Dict]:
        """Load raw dataset from JSON file.
        
        Args:
            path: Path to the dataset file
            
        Returns:
            List of dictionary items from the dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with path.open() as f:
            return json.load(f)

    @staticmethod
    def build_triplets(raw: List[Dict], negative_field: str) -> Dataset:
        """Create a dataset with (anchor, positive, negative) triplets.
        
        Args:
            raw: Raw dataset as a list of dictionaries
            negative_field: Field name to use for negative examples
            
        Returns:
            Dataset with three columns: anchor, positive, negative
        """
        anchors, positives, negatives = [], [], []

        for row in raw:

            if negative_field not in row:
                continue

            anchors.append(row["word"])

            positives.append(row["definition"])

            negatives.append(row[negative_field])

        return Dataset.from_dict({

            "anchor": anchors,

            "positive": positives,

            "negative": negatives,

        })

    @staticmethod
    def concat_and_split(
        datasets: List[Dataset], 
        train_frac: float = CONFIG["TRAIN_FRAC"], 
        seed: int = CONFIG["SEED"]
    ) -> DatasetDict:
        """Concatenate multiple datasets and split into train/eval.
        
        Args:
            datasets: List of datasets to concatenate
            train_frac: Fraction of data to use for training
            seed: Random seed for reproducibility
            
        Returns:
            DatasetDict with 'train' and 'eval' splits
        """
        merged = concatenate_datasets(datasets)

        split = merged.shuffle(seed=seed).train_test_split(train_size=train_frac, seed=seed)

        return DatasetDict({

            "train": split["train"],

            "eval": split["test"],

        })