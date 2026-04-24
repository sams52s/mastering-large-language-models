"""
Module containing LaplaceLanguageModel class - an extension of NGramLanguageModel.
"""
from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

from collections import Counter, defaultdict
from typing import List, Tuple, Dict

from LanguageModeling.task01_ngrams.task import NGramLanguageModel, EOS, UNK

class LaplaceLanguageModel(NGramLanguageModel): 
    """ 
    An extended version of NGramLanguageModel that implements Laplace smoothing.
    """
    def __init__(self, lines: List[str], n: int, delta: float = 1.0):
        """
        Initialize the LaplaceLanguageModel object with the given parameters.

        :param lines: List of text lines.
        :param n: Size of n-grams.
        :param delta: Smoothing parameter.

        Note: You don't need to store all the tokens in self.probs. Store only tokens that have been seen after the given prefix.
        """
        self.n = n
        counts = self.count_ngrams(lines, self.n)
        self.vocab = set(token for token_counts in counts.values() for token in token_counts)
        self.probs = defaultdict(lambda: defaultdict(float))

        # TODO: initialize self.probs
            
    def get_possible_next_tokens(self, prefix: Tuple[str, ...]) -> Dict[str, float]:
        """
        Returns possible next tokens and their probabilities given a prefix.

        :param prefix: Prefix tuple.
        :return: Dictionary with tokens and their probabilities.

        Note: Missing tokens should have uniform probability among all missing tokens.
        Note 2: It's a design choice not to store all the tokens in self.probs, therefore,
            we need to calculate missing tokens probabilities on the fly.
        """
        token_probs = super().get_possible_next_tokens(prefix)
        missing_prob_total = 1.0 - sum(token_probs.values())
        missing_prob = missing_prob_total / max(1, len(self.vocab) - len(token_probs))
        return {token: token_probs.get(token, missing_prob) for token in self.vocab}
    
    def get_next_token_prob(self, prefix: Tuple[str, ...], next_token: str) -> float:
        """
        Returns the probability of a specific next token given a prefix.

        :param prefix: Prefix tuple.
        :param next_token: Next token.
        :return: Probability of the next token.

        Note: we're choosing to assign the missing probability even to the tokens we haven't seen in the training data.
        """
        token_probs = super().get_possible_next_tokens(prefix)
        if next_token in token_probs:
            return token_probs[next_token]
        else:
            missing_prob_total = 1.0 - sum(token_probs.values())
            missing_prob_total = max(0, missing_prob_total)  # prevent rounding errors
            return missing_prob_total / max(1, len(self.vocab) - len(token_probs))


def main():
    sample_lines = ["this is a sample sentence", "another example for testing"]
    model = LaplaceLanguageModel(sample_lines, n=2, delta=0.5)
    print("Laplace-smoothed model created with delta=0.5")
    print("Probability of 'is' after 'this':", model.get_next_token_prob("this", "is"))
    print("Probability of unknown token after 'this':", model.get_next_token_prob("this", "unknown"))


if __name__ == '__main__':
    main()