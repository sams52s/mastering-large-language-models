import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from MinLlama.HelperCode.config import download_data
from MinLlama.Llama.task import load_pretrained

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    sanity_data_path = os.path.join(os.path.dirname(__file__), "sanity_check.data")
    stories_model_path = os.path.join(os.path.dirname(__file__), "stories42M.pt")

    if os.path.exists(sanity_data_path):
        os.remove(sanity_data_path)
    download_data("https://raw.githubusercontent.com/neubig/minllama-assignment/master/sanity_check.data",
                  sanity_data_path)
    sanity_data = torch.load(sanity_data_path, weights_only=False)

    # text_batch = ["hello world", "hello neural network for NLP"]
    # tokenizer here
    sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                             [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])

    if os.path.exists(stories_model_path):
        os.remove(stories_model_path)
    download_data("https://www.cs.cmu.edu/~vijayv/stories42M.pt", stories_model_path)
    # load our model
    llama = load_pretrained(stories_model_path)
    with torch.no_grad():
        logits, hidden_states = llama(sent_ids)
        assert torch.allclose(logits, sanity_data["logits"], atol=1e-4, rtol=1e-2)
        assert torch.allclose(hidden_states, sanity_data["hidden_states"], atol=1e-4, rtol=1e-2)
        print("Your Llama implementation is correct!")
