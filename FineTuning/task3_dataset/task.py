from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import Tuple
from datasets import load_from_disk

class DatasetHandler:
    """Handler for processing and managing datasets"""
    TEMPLATE = f"""\
    word: {{word}}
    definition: {{definition}}
    example: {{example}}
    """.replace("    ", "").strip()
    def __init__(self, tokenizer_name: str, num_proc: int = 1) -> None:
        """Initialize the handler with the tokenizer name
        :@param tokenizer_name: Hugging Face tokenizer name
        :@param num_proc: Number of processes to use for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.num_proc = num_proc    

    def convert_to_hf(self, json_path: str) -> Dataset:
        """Convert JSON dataset to Hugging Face Dataset format"""
        with open(json_path) as f:
            data = json.load(f)
        
        return Dataset.from_list(data)
    
    def _add_text_column(self, dataset: Dataset) -> Dataset:
        """Add a text column to the dataset
        
        Format: Use the TEMPLATE to combine word, definition, and example
        """
        dataset = # TODO: add "text" column. note: use self.num_proc
        return dataset
    
    def process(self, dataset: Dataset, text_col: str = "text") -> Dataset:
        """Add text column and tokenize along it
        
        Note: No truncation is applied since the text is short, no padding is needed since we'll use collate_fn
        """
        dataset = self._add_text_column(dataset)
        
        # TODO: tokenize the texts and leave only attention mask and input ids in the dataset (both are the result of tokenizer application)
    
    def train_test_split(self, dataset: Dataset, test_size: float = 0.15, random_state: int = 42) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and test sets"""
        split_data = dataset.train_test_split(test_size=test_size, seed=random_state)
        return split_data['train'], split_data['test']
    
    def save(self, train: Dataset, test: Dataset, save_dir: str) -> None:
        """Save train and test datasets to disk"""
        train.save_to_disk(f"{save_dir}/train")
        test.save_to_disk(f"{save_dir}/test")
    
    @staticmethod
    def load(load_dir: str) -> Tuple[Dataset, Dataset]:
        """Load train and test datasets from disk"""
        return load_from_disk(f"{load_dir}/train"), load_from_disk(f"{load_dir}/test")
