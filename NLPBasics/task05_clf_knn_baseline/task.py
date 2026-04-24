# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import numpy as np
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config
from tools_basics.data_handler import DataHandler
# noinspection PyUnresolvedReferences
from task03_glove_embeddings.task import GloVeEmbeddings


class KNNClassifier:
    """A k-Nearest Neighbors (kNN) classifier based on cosine similarity."""

    def __init__(self, train_df, test_df, get_phrase_embedding_fn):
        self.train_df = train_df
        self.test_df = test_df
        self.get_phrase_embedding = get_phrase_embedding_fn
        self.X_train = np.array([self.get_phrase_embedding(phrase) for phrase in self.train_df.text])
        self.y_train = self.train_df.label.values

    def cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the cosine similarity between two vectors."""
        pass # TODO: compute cosine similarity. Note: if any vector close to zero, return zero

    def find_nearest(self, query: str, k: int = 10) -> tuple:
        """
        Given a text query, return k most similar lines from the training data.
        Similarity is measured using cosine similarity between phrase embeddings.

        Steps:
        1. Compute cosine similarity between the query and all training samples.
        2. Find top_k_sorted_ids: indices of the k most similar training samples to the query (sorted by similarity).
            * Hint: use np.argpartition and np.argsort
        3. Return the text and labels of the top k most similar training samples.
        """
        k = min(k, len(self.train_df))  # Ensure k is not larger than the training set size
        query_vec = self.get_phrase_embedding(query)

        # TODO: get top_k_sorted_ids

        return self.train_df.iloc[top_k_sorted_ids].text.values, self.y_train[top_k_sorted_ids]

    def get_accuracy(self, X_test_phrases: list, y_test: np.ndarray, k: int) -> float:
        """Compute the accuracy of the kNN classifier on the test dataset.
        
        Steps:
            1. For each test phrase, find the k most similar training samples.
            2. Extract the prediction, that is the most common label among the k nearest (e.g. [0, 1, 1, 0, 1] -> 1).
                Note: we assume k is odd to avoid ties.
                Hint: you can do it through np.bincount or np.sum
        """
        correct = 0
        for i, phrase in tqdm(enumerate(X_test_phrases), total=len(X_test_phrases)):
            # TODO: extract pred for given phrase
            correct += pred == y_test[i]
        return correct / len(y_test)

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    dh = DataHandler(conf)
    train_df, test_df = dh.get_data()

    glove = GloVeEmbeddings() # ~1.5min for 100d
    knn_classifier = KNNClassifier(train_df, test_df, glove.get_phrase_embedding)

    # Sample and truncate test data for faster execution
    n_samples = 100
    trunc_test_df = test_df.sample(n_samples)
    X_test_phrases_trunc = trunc_test_df.text.values
    y_test_trunc = trunc_test_df.label.values
    
    accuracy = knn_classifier.get_accuracy(X_test_phrases_trunc, y_test_trunc, k=3)
    print(f"k-NN Classifier Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
