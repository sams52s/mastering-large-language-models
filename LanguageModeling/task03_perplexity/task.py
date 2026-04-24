from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up = 2, verbose = True)

import numpy as np
from typing import List

from LanguageModeling.task01_ngrams.task import BaseLanguageModel, EOS

class Evaluator:
    """
    Class for evaluating language model perplexity.
    """

    @staticmethod
    def perplexity(model: BaseLanguageModel, lines: List[str], min_logprob: float = np.log(10**-50.)) -> float:
        """
        Calculate the perplexity of the language model on a given corpus.

        :param model: BaseLanguageModel object representing the language model.
        :param lines: List of strings with space-separated tokens.
        :param min_logprob: Minimum log probability threshold.
            Probability of the next token will be taken into account as max(min_logprob, log(P(token|prefix)) )
        :return: Calculated perplexity value.
        
        Note: do not forget to compute P(w_first | empty) and P(eos | full_sequence)
        
        PLEASE USE model.get_next_token_prob and NOT model.get_possible_next_tokens
        """
        total_logprob, total_num_tokens = 0, 0
        for line in lines:
            tokens = line.split()
            tokens.append(EOS)
            pref = ''
            total_num_tokens += len(tokens)

            for token in tokens:
                prob = model.get_next_token_prob(pref, token)
                log_prob = np.log(prob) if prob > 0 else min_logprob
                total_logprob += max(min_logprob, log_prob)
                pref = pref + ' ' + token  # ← no .strip(), keeps leading space

        return np.exp(-total_logprob / total_num_tokens)


def main():
    from LanguageModeling.task01_ngrams.task import NGramLanguageModel
    sample_lines = ["this is a sample sentence", "another example for testing"]
    test_lines = ["this is a test"]
    model = NGramLanguageModel(sample_lines, n=2)
    perplexity = Evaluator.perplexity(model, test_lines)
    print(f"Model perplexity on test data: {perplexity:.2f}")


if __name__ == '__main__':
    main()