# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import pandas as pd
from nltk.tokenize import WordPunctTokenizer

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config
from tools_basics.data_handler import DataHandler


class Tokenizer:
    """Class to tokenize the text data."""
    
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the input text.
        
        1. Lowercase the text
        2. Remove '<br />' tags
        3. Tokenize the text using WordPunctTokenizer
        """
        # TODO: implement tokenize function

        text = text.lower()
        text = text.replace('<br />', '')
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """Process the text (tokenize and join)."""
        pass # TODO: YOUR CODE HERE

        tokens = self.tokenize(text)
        return ' '.join(tokens)

    def apply_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies preprocessing of the 'text' column of the DataFrame.
        
        Use `pd.Series.apply`
        """
        # TODO: YOUR CODE HERE

        df['text'] = df['text'].apply(self.preprocess_text)
        return df

def get_sample_texts(df: pd.DataFrame, n: int = 3, random_state: int = 42) -> list:
    """Return a sample texts from the DataFrame."""
    return df.sample(n, random_state=random_state)['text'].tolist()

def print_texts(texts: list, title: str, n_truncate: int = 100) -> None:
    """Print the sample texts."""
    print(title)
    for text in texts:
        print(text[:n_truncate])
    print()

def run_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = True) -> tuple:
    """Run preprocessing on the train and test DataFrames."""
    if verbose:
        sample_before = get_sample_texts(train_df)
        print_texts(sample_before, "Sample texts before tokenization:")
    
    tokenizer = Tokenizer()
    train_df = tokenizer.apply_preprocess(train_df)
    test_df = tokenizer.apply_preprocess(test_df)
    
    if verbose:
        sample_after = get_sample_texts(train_df)
        print_texts(sample_after, "Sample texts after tokenization:")
    
    return train_df, test_df

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    dh = DataHandler(conf)
    train_df, test_df = dh.get_data()
    train_df, test_df = run_preprocessing(train_df, test_df)
    
    # Save the tokenized data
    dh.save_data(train_df, test_df)

if __name__ == '__main__':
    main()
