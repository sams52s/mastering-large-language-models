"""
Module with implementation of RNNLanguageModel class (Recurrent Neural Network Language Model).
"""

from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import torch
from torch import nn
from LanguageModeling.task01_ngrams.task import BaseLanguageModel, EOS, UNK
from LanguageModeling.task05_text_tools.task import TextTools as tt

class RNNLanguageModel(nn.Module, BaseLanguageModel):
    def __init__(self, tokens: list, emb_size: int = 16, hid_size: int = 256):
        """ 
        Build a recurrent language model.
        You are free to choose anything you want, but the recommended architecture is
        - token embeddings
        - one or more LSTM/GRU layers with hid size
        - linear layer to predict logits
        
        :param tokens: List of tokens.
        :param emb_size: Size of the embedding layer.
        :param hid_size: Size of the hidden layer.
        :note: if you use nn.RNN/GRU/LSTM, make sure you specify batch_first=True
         With batch_first, your model operates with tensors of shape [batch_size, sequence_length, num_units]
         Also, please read the docs carefully: they don't just return what you want them to return :)
        """
        super().__init__() # initialize base class to track sub-layers, trainable variables, etc.
        
        n_tokens = len(tokens)
        self.tokens = tokens

        # Important: You can experiment with the architecture
        self.emb = nn.Embedding(n_tokens, emb_size)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size, batch_first=True)
        self.linear = nn.Linear(hid_size, n_tokens)

    def __call__(self, input_ix: torch.Tensor) -> torch.Tensor:
        """
        compute language model logits given input tokens
        :param input_ix: batch of sequences with token indices, tensor: int32[batch_size, sequence_length]
        :returns: pre-softmax linear outputs of language model [batch_size, sequence_length, n_tokens]
            these outputs will be used as logits to compute P(x_t | x_0, ..., x_{t - 1})
        """
        embeds = self.emb(input_ix)  # [batch, seq_len, emb_size]
        rnn_out, _ = self.rnn(embeds)  # [batch, seq_len, hid_size]
        logits = self.linear(rnn_out)  # [batch, seq_len, n_tokens]
        return logits
    
    def get_possible_next_tokens(self, prefix: str) -> dict:
        """ 
        :returns: probabilities of next token, dict {token : prob} for all tokens

        Note: Use torch.no_grad
        Steps:
        1. Convert to matrix
        2. Get models output
        3. Apply softmax and return the result
        """
        with torch.no_grad():
            input_ix = tt.to_matrix([prefix])          # [1, seq_len]
            logits = self(input_ix)                     # [1, seq_len, n_tokens]
            probs = torch.softmax(logits[0, -1, :], dim=-1)  # [n_tokens]
        return {token: probs[i].item() for i, token in enumerate(self.tokens)}
    
    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """ :returns: probability of next_token given prefix, float """
        return self.get_possible_next_tokens(prefix).get(next_token, 0.0)


def main():
    from LanguageModeling.task05_text_tools.task import TextTools as tt
    tokens = tt.TOKENS
    model = RNNLanguageModel(tokens, emb_size=8, hid_size=32)
    print(f"Created RNN model with {sum(p.numel() for p in model.parameters())} parameters")
    sample_text = "Hello"
    print(f"Top 3 next tokens for '{sample_text}':", 
          sorted(model.get_possible_next_tokens(sample_text).items(), key=lambda x: x[1], reverse=True)[:3])


if __name__ == '__main__':
    main()