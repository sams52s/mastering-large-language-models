# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
# custom gensim data directory
os.environ['GENSIM_DATA_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), './.gensim_data')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import os
import numpy as np
import gensim.downloader as api

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config, save_as_numpy
from tools_basics.data_handler import DataHandler
# noinspection PyUnresolvedReferences
from task02_tokenization.task import Tokenizer

class GloVeEmbeddings:
    """Class to load and work with GloVe embeddings."""
    
    def __init__(self, model_name: str = 'glove-twitter-25') -> None:
        """Initialize the class and load the GloVe model."""
        self.model = api.load(model_name)
        self.tokenizer = Tokenizer()
        self._unknown_emb = np.zeros(self.model.vector_size)

    def get_word_vectors(self, words: list):
        """Get the vector representations for a list of words.
        
        Note: In case a word is not in the vocabulary, return a zero vector."""
        pass # TODO: use model.get_vector
        vectors = []
        for word in words:
            if word in self.model:
                vectors.append(self.model.get_vector(word))
            else:
                vectors.append(self._unknown_emb)
        return np.array(vectors)
    
    def get_phrase_embedding(self, phrase: str) -> np.ndarray:
        """
        Convert phrase to a vector by aggregating word embeddings.
        - Lowercase the phrase
        - Tokenize the phrase
        - Average the word vectors for all words in the tokenized phrase
        - Skip words not in the model's vocabulary
        - If all words are missing from the vocabulary, return zeros
        """
        pass # TODO: YOUR CODE HERE
        tokens = self.tokenizer.tokenize(phrase)
        vectors = [self.model.get_vector(token) for token in tokens if token in self.model]

        if len(vectors) == 0:
            return self._unknown_emb

        return np.mean(vectors, axis=0)

    def compute_phrase_vectors(self, phrases: list[str], max_tokens: int | None = 30) -> np.ndarray:
        """Truncate and compute vectors for a list of phrases.
        
        Args:
            phrases (list[str]): List of phrases to compute embeddings for.
            max_tokens (int): Maximum number of tokens to consider in each phrase. (If None, consider all tokens)
        """
        pass # TODO: YOUR CODE HERE
        phrase_vectors = []
        for phrase in phrases:
            if max_tokens is not None:
                tokens = self.tokenizer.tokenize(phrase)
                phrase = " ".join(tokens[:max_tokens])

            phrase_vectors.append(self.get_phrase_embedding(phrase))

        return np.array(phrase_vectors)


N_WORDS = 1000
N_PHRASES = 1000
def extract_and_save_data(train_df, test_df, glove, conf):
    """Extract data for the following tasks:
    1. Words visualization (word embeddings, words)
    2. Chosen phrases visualization (phrase embeddings, phrases)
    3. Train and test data embeddings (train_embeddings, test_embeddings, target)
    
    Then, save
    """
    words = glove.model.index_to_key[:N_WORDS]
    word_vectors = glove.get_word_vectors(words)

    phrases = train_df.sample(N_PHRASES, random_state=42).text.tolist()
    phrase_vectors = glove.compute_phrase_vectors(phrases)

    train_embeddings = glove.compute_phrase_vectors(train_df.text.tolist(), max_tokens=None)
    train_target = train_df.label.values
    test_embeddings = glove.compute_phrase_vectors(test_df.text.tolist(), max_tokens=None)
    test_target = test_df.label.values

    # Save
    for data, path in (
        (words, conf.path.words),
        (word_vectors, conf.path.word_embs),
        (phrases, conf.path.phrases),
        (phrase_vectors, conf.path.phrase_embs),
        (train_embeddings, conf.path.train_emb),
        (train_target, conf.path.train_target),
        (test_embeddings, conf.path.test_emb),
        (test_target, conf.path.test_target),
    ):
        save_as_numpy(data, path)
    

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    dh = DataHandler(conf)
    train_df, test_df = dh.get_data()

    glove = GloVeEmbeddings() # ~1.5min for 100d

    extract_and_save_data(train_df, test_df, glove, conf)
    

if __name__ == '__main__':
    main()
