"""
Module with implementation of Generator class for sequence token generation.
"""
from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

from LanguageModeling.task01_ngrams.task import BaseLanguageModel, UNK, EOS
from typing import List
import numpy as np
from functools import partial


class Generator:
    """
    Class for generating token sequences based on a language model.
    """
    def __init__(self, model: BaseLanguageModel, token_level: str = 'word') -> None:
        """
        Initialize the Generator object.
        
        :param model: an instance of BaseLanguageModel
        :param token_level: either 'word' or 'char'
        """
        assert token_level in ['word', 'char'], f"Unknown token_level: {token_level}"
        self.model = model

        if token_level == 'word':
            self.tokenize = lambda x: x.split()
            self.detokenize = lambda x: ' '.join(x)
        elif token_level == 'char':
            self.tokenize = list
            self.detokenize = lambda x: ''.join(x)

    def _get_tokens_probs_safe(self, prefix: str) -> dict[str, float]:
        """
        Get tokens and their probabilities for a given prefix.
        In case of an empty list of tokens, return EOS with probability 1.
        
        :param prefix: Prefix of the sequence.
        :return: Dictionary with tokens and their probabilities.
        """
        tokens_probs = self.model.get_possible_next_tokens(prefix)
        if not tokens_probs:
            tokens_probs[EOS] = 1
        return tokens_probs

    def get_next_token_sample(self, prefix: str, temperature: float = 1.0) -> str:
        """
        Return next token after prefix.
        
        :param prefix: Prefix of the sequence.
        :param temperature: samples proportionally to lm probabilities ^ (1 / temperature)
            if temperature == 0, always takes most likely token (greedy). Break ties arbitrarily.
        :return: Predicted next token.
        """
        probs = self._get_tokens_probs_safe(prefix)
        # TODO: temperature == 0 case
        
        pass # TODO: general case
    
    def get_next_token_nucleus(self, prefix: str, nucleus: float = 0.9) -> str:
        """
        Generate a sequence with nucleus sampling.
        
        :param prefix: Prefix of the sequence.
        :param nucleus: N from the formulae above, N \in [0, 1]
        :return: Predicted next token.
        :note: make sure that nucleus always contains at least one word, even if p(w*) > nucleus

        Steps:
        1. Get sorted_probs_ids (indices of tokens sorted by probabilities in descending order) and sorted_probs (sorted probabilities)
        2. Calculate cumulative probabilities cum_probs and create a mask for the tokens that should be included in the nucleus
            Note: if no token should be included, include the most probable one (that is mask[0] is always True)
        3. Normalize the probabilities of the selected tokens and sample the next token
        4. Return the sampled token
        """
        token_probs = self._get_tokens_probs_safe(prefix)
        tokens, probs = zip(*token_probs.items())
        assert np.isclose(sum(probs), 1), f"Sum of probabilities is not close to 1: {sum(probs)}"

        # TODO: Step 1
        # TODO: Step 2
        
        # TODO: Step 3
        return next_token
    
    def generate_sequence(self, prefix: str = UNK, mode: str = 'sample', max_len: int = 100, **kwargs) -> List[str]:
        """
        Generate a sequence of tokens.
        
        :param prefix: Prefix of the sequence.
        :param mode: either 'sample' or 'nucleus'
        :param **kwargs: additional arguments for the chosen mode
            temperature: for mode='sample'
            nucleus, max_len: for mode='nucleus'
        :returns: A list of tokens (including EOS)
        """
        assert mode in ['sample', 'nucleus'], f"Unknown mode: {mode}"
        if mode == 'sample':
            temperature = kwargs.get('temperature', 0.5)
        elif mode == 'nucleus':
            nucleus = kwargs.get('nucleus', 0.9)
        
        sequence = self.tokenize(prefix)
        if mode == 'sample':
            sample_fn = partial(self.get_next_token_sample, temperature=temperature)
        elif mode == 'nucleus':
            sample_fn = partial(self.get_next_token_nucleus, nucleus=nucleus)
        
        while True:
            pass # TODO: Generate next token. Note: Stop if EOS or max_len is reached
        return sequence


def main():
    from LanguageModeling.task01_ngrams.task import NGramLanguageModel
    sample_lines = ["this is a sample sentence", "another example for testing"]
    model = NGramLanguageModel(sample_lines, n=2)
    generator = Generator(model)
    generated = generator.generate_sequence(max_len=10)
    print("Generated sequence:", generator.detokenize(generated))

    # with prefix 'another'
    generated = generator.generate_sequence(prefix='another', max_len=10)
    print("Generated sequence with prefix 'another':", generator.detokenize(generated))


if __name__ == '__main__':
    main()