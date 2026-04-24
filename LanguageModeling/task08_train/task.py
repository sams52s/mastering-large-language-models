"""
Module for training procedure class. 
"""
from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import os
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from einops import rearrange
from typing import List, Tuple

from LanguageModeling.task05_text_tools.task import TextTools as tt
from LanguageModeling.task02_generation.task import Generator
from LanguageModeling.task06_rnn.task import RNNLanguageModel
from LanguageModeling.task07_loss.task import CrossEntropyLoss


class TrainProcedure:
    """
    Class for training the model and computing losses.
    """
    @staticmethod
    def _compute_loss(model: torch.nn.Module, loss_fn, lines: List[str], batch_ids: List[int], device: torch.device) -> torch.Tensor:
        """
        Compute loss for the model.
        
        :param model: Model for which the loss is computed.
        :param loss_fn: Loss function.
        :param lines: List of lines.
        :param batch_ids: List of batch IDs.
        :param device: Device to move tensors to.
        :return: Computed loss.
        """
        input_idx = tt.to_matrix([lines[i] for i in batch_ids]).to(device)
        logits = model(input_idx)  # [batch_size, len, vocab]
        logits = rearrange(logits, 'b l v -> b v l')
        loss = loss_fn(logits, input_idx)
        return loss

    @staticmethod
    def score_lines(model: torch.nn.Module, lines: List[str], loss_fn, batch_size: int, device: torch.device) -> float:
        """
        Computes average loss over the entire dataset.
        
        :param model: Model for which the loss is computed.
        :param lines: List of lines.
        :param loss_fn: Loss function.
        :param batch_size: Size of the batch.
        :param device: Device to move tensors to.
        :return: Average loss.
        """
        all_losses = []
        for i in range(0, len(lines), batch_size):
            with torch.no_grad():
                batch_ids = range(i, min(i + batch_size, len(lines)))
                loss = TrainProcedure._compute_loss(model, loss_fn, lines, batch_ids, device)
                all_losses.append(loss.item())
        return np.mean(all_losses)

    @staticmethod
    def train(model: torch.nn.Module, opt, loss_fn, train_lines: List[str], dev_lines: List[str], device: torch.device, gen_conf: dict, batch_size: int = 64, draw_every: int = 50, score_dev_every: int = 250, n_epochs: int = 5000) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]], torch.nn.Module]:
        """
        Train the model using the training lines and evaluate on development lines.
        
        :param model: Model to be trained.
        :param opt: Optimizer for training.
        :param loss_fn: Loss function for optimization.
        :param train_lines: List of training lines.
        :param dev_lines: List of development lines.
        :param device: Device to move tensors to.
        :param gen_conf: Configuration for the generator.
        :param batch_size: Size of the batch.
        :param draw_every: Frequency at which to draw loss plot.
        :param score_dev_every: Frequency at which to score development set.
        :param n_epochs: Number of epochs to train for.
        :return: Training history, development history, and trained model.
        """
        train_history, dev_history = [], []
        model = model.to(device)
        gen = Generator(model, token_level='char')  # TODO: is model mutable? 
        for i in trange(n_epochs):
            batch_ids = np.random.choice(len(train_lines), batch_size, replace=False)
            loss_i = TrainProcedure._compute_loss(model, loss_fn, train_lines, batch_ids, device)
            
            loss_i.backward()
            opt.step()
            opt.zero_grad()
            
            train_history.append((i, loss_i.item()))
            
            if (i + 1) % draw_every == 0:
                TrainProcedure.produce_meta(train_history, dev_history, gen, gen_conf)
            
            if (i + 1) % score_dev_every == 0:
                print("Scoring dev...")
                dev_history.append((i, TrainProcedure.score_lines(model, dev_lines, loss_fn, batch_size, device)))
                print('#%i Dev loss: %.3f' % dev_history[-1])
        
        return train_history, dev_history, model
    
    @staticmethod
    def produce_meta(train_history: List[Tuple[int, float]], dev_history: List[Tuple[int, float]], gen: Generator, gen_conf: dict):
        """
        Produce meta information during training.
        
        :param train_history: Training history.
        :param dev_history: Development history.
        :param gen: Generator object.
        :param gen_conf: Configuration for the generator.
        """
        # plt.clf()
        plt.figure(figsize=[12, 4])
        plt.scatter(*zip(*train_history), alpha=0.1, label='train_loss')
        if len(dev_history):
            plt.plot(*zip(*dev_history), color='red', label='dev_loss')
        path_to_save = os.path.join(os.path.dirname(__file__), 'train_test_plot.png')
        plt.legend(); plt.grid(); plt.savefig(path_to_save)

        gen_conf_str = ', '.join([f'{k}={v}' for k, v in gen_conf.items()])
        print(f"Generated examples for ({gen_conf_str}):")
        for _ in range(3):
            seq = gen.generate_sequence(**gen_conf)
            print(''.join(seq))


def main():
    # Create tiny dummy data for demonstration
    sample_lines = ["Hello world", "This is a test"]
    dev_lines = ["Hello test"]
    
    # Create model and optimizer
    model = RNNLanguageModel(tt.TOKENS, emb_size=8, hid_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()
    device = torch.device("cpu")
    
    # Train for just a few steps
    gen_conf = {"prefix": "H", "max_len": 10, "mode": "sample", "temperature": 0.8}
    train_history, dev_history, trained_model = TrainProcedure.train(
        model, optimizer, loss_fn, sample_lines, dev_lines, device, gen_conf, 
        batch_size=2, draw_every=2, score_dev_every=2, n_epochs=4
    )
    
    print("Completed tiny training demonstration")


if __name__ == '__main__':
    main()