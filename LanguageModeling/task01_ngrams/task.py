"""
Module with implementation of APIBase class (Base class for API classes).
"""
from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

from collections import defaultdict, Counter
from abc import ABC, abstractmethod
from typing import Dict, Tuple, DefaultDict, Iterable

BOS, EOS = "→", "←"  # one-letter tokens (not the best practice)
UNK = BOS
class BaseLanguageModel(ABC):
    """
    Abstract base class for language models.
    """
    BOS = BOS
    EOS = EOS
    UNK = UNK

    @abstractmethod
    def get_possible_next_tokens(self, prefix: str) -> Dict[str, float]:
        """
        Get possible next tokens and their probabilities given a prefix.

        :param prefix: string with space-separated prefix tokens
        :return: Dictionary {token : probability} for all tokens with positive probabilities
        """
        pass

    @abstractmethod
    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """
        Get probability of the next token given a prefix.

        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :return: Probability of next_token given prefix, a value between 0 and 1
        """
        pass


class NGramLanguageModel(BaseLanguageModel):
    """
    Class for a simple n-gram language model.
    """
    @staticmethod
    def count_ngrams(lines: Iterable[str], n: int) -> DefaultDict[Tuple[str, ...], Counter[str]]:
        """
        Count n-grams occurrences in the lines.

        :param lines: an iterable of strings with space-separated tokens
        :param n: n-gram size
        :return: Dictionary { tuple(prefix_tokens): {next_token_1: count_1, next_token_2: count_2}}

        When building counts, please consider the following two edge cases:
        - if prefix is shorter than (n - 1) tokens, it should be padded with UNK. For n=3,
        empty prefix: "" -> (UNK, UNK)
        short prefix: "the" -> (UNK, the)
        long prefix: "the new approach" -> (new, approach)
        - you should add a special token, EOS, at the end of each sequence
        "... with deep neural networks ." -> (..., with, deep, neural, networks, ., EOS)
        count the probability of this token just like all others.
        """
        counts: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)

        # TODO: fill in the counts
        prefix_size = n - 1
        for line in lines:
            tokens = line.split() + [EOS]
            history = [UNK] * prefix_size

            for token in tokens:
                prefix = tuple(history[-prefix_size:]) if prefix_size > 0 else tuple()
                counts[prefix][token] += 1
                if prefix_size > 0:
                    history.append(token)


        return counts

    def __init__(self, lines: Iterable[str], n: int) -> None:
        """
        Initialize the n-gram language model.

        :param n: n-gram size
        :param lines: an iterable of strings with space-separated tokens
        """
        assert n >= 1
        self.n: int = n

        counts = self.count_ngrams(lines, self.n)

        # compute token probabilities given counts
        self.probs: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)

        # populate self.probs with actual probabilities
        for pref, cnt in counts.items():
            total = sum(cnt.values())
            for token, count in cnt.items():
                self.probs[pref][token] = count / total

    def get_possible_next_tokens(self, prefix: str) -> Dict[str, float]:
        """
        Get possible next tokens and their probabilities given a prefix.

        :param prefix: string with space-separated prefix tokens
        :return: Dictionary {token : probability} for all tokens with positive probabilities

        Steps:
        1. Complete or truncate the prefix if needed (note: for completion add UNK tokens in the beginning)
        2. The the distribution of tokens given prefix (use self.probs)
        """
        prefix_tokens = prefix.split()
        prefix_size = self.n - 1

        if prefix_size == 0:
            normalized_prefix = tuple()
        elif len(prefix_tokens) < prefix_size:
            normalized_prefix = tuple([UNK] * (prefix_size - len(prefix_tokens)) + prefix_tokens)
        else:
            normalized_prefix = tuple(prefix_tokens[-prefix_size:])

        return dict(self.probs.get(normalized_prefix, {}))


    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """
        Get probability of the next token given a prefix.

        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :return: Probability of next_token given prefix, a value between 0 and 1
        """
        return self.get_possible_next_tokens(prefix).get(next_token, 0)


def main():
    sample_lines = ["this is a sample sentence", "another example for testing"]
    model = NGramLanguageModel(sample_lines, n=2)
    print("Sample model created with bigrams")
    print("Possible next tokens after 'this':", model.get_possible_next_tokens("this"))


if __name__ == '__main__':
    main()