import torch

from MinLlama.HelperCode.config import download_data
from MinLlama.Llama.task import load_pretrained

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    download_data("https://raw.githubusercontent.com/jetbrains-academy/llm-agent-course/main/MinLlama/SanityCheck/sanity_check.data",
                  "sanity_check.data")
    sanity_data = torch.load("./sanity_check.data")
    # text_batch = ["hello world", "hello neural network for NLP"]
    # tokenizer here
    sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                             [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])

    download_data("https://www.cs.cmu.edu/~vijayv/stories42M.pt", "stories42M.pt")

    # load our model
    llama = load_pretrained("./stories42M.pt")
    with torch.no_grad():
        logits, hidden_states = llama(sent_ids)
        assert torch.allclose(logits, sanity_data["logits"], atol=1e-4, rtol=1e-2)
        assert torch.allclose(hidden_states, sanity_data["hidden_states"], atol=1e-4, rtol=1e-2)
        print("Your Llama implementation is correct!")
