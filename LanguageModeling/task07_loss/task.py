"""
Module for computing loss functions for language models.
"""
from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from LanguageModeling.task05_text_tools.task import TextTools as tt


class CrossEntropyLoss(nn.Module):
    """
    Module for computing cross-entropy loss function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, input_idx: torch.Tensor) -> torch.Tensor:
        """
        Note: nn.CrossEntropyLoss input: 
        logits -- (N, C, d1, d2, ..., dK) where N is batch size and C is number of classes
        target -- (N, d1, d2, ..., dK) where each value is 0 <= targets[i] <= C-1
        Compute the cross-entropy loss over non-eos tokens.
        
        Your task: implement loss function as per formula above
        your loss should only be computed on actual tokens, excluding padding
        predicting actual tokens and first EOS do count. Subsequent EOS-es don't
        you may or may not want to use the compute_mask function from above.
        
        :param logits: Float32 tensor of shape [batch, n_tokens, seq_len].
        :param input_idx: Int32 tensor of shape [batch, seq_len] with answers.
        :returns: Scalar float32 loss function (mean crossentropy over non-eos tokens).

        Please, use tt.compute_mask function to get masks for input_idx.
        :note: Avoid using loops in your code (it will significantly slow down training)

        Steps:
        1. Compute the mask for input_idx.
            Note: Don't forget to exclude BOS token from the mask. (i.e. mask[:, 1:])
        2. Flatten the logits and input_idx tensors.
            Note: Exclude the last token from each sequence (possible EOS).
        3. Get target values from input_idx tensor and flatten them. (Again, exclude BOS token)
        4. Extract valid logits and references using the computed mask.
            Note: "valid" are ones which we want to see in the loss (that is all tokens after BOS (exclusive) and before EOS (inclusive)).
        5. Compute the cross-entropy loss using F.cross_entropy function.
        6. Return the computed loss.
        """
        device = logits.device
        masks = tt.compute_mask(input_idx).to(device)
        mask = masks[:, 1:]  # [batch, seq_len-1]

        # Step 2: flatten logits, exclude last position
        # logits: [batch, n_tokens, seq_len] → drop last time step → rearrange
        flat_logits = rearrange(logits[:, :, :-1], 'b c t -> (b t) c')  # [batch*(seq_len-1), n_tokens]

        # Step 3: targets = input_idx excluding BOS, flattened
        flat_targets = rearrange(input_idx[:, 1:], 'b t -> (b t)')  # [batch*(seq_len-1)]

        # Step 4: apply mask to keep only valid positions
        flat_mask = rearrange(mask, 'b t -> (b t)')  # [batch*(seq_len-1)]
        valid_logits = flat_logits[flat_mask]  # [n_valid, n_tokens]
        valid_targets = flat_targets[flat_mask]  # [n_valid]

        # Steps 5 & 6: compute and return loss
        return F.cross_entropy(valid_logits, valid_targets)

def main():
    import torch
    from LanguageModeling.task05_text_tools.task import TextTools as tt
    
    loss_fn = CrossEntropyLoss()
    batch_size, seq_len, vocab_size = 2, 5, 10
    
    # Create dummy data
    logits = torch.randn(batch_size, vocab_size, seq_len)  # [batch, vocab, seq_len]
    input_idx = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch, seq_len]
    
    loss = loss_fn(logits, input_idx)
    print(f"Computed loss on dummy data: {loss.item():.4f}")


if __name__ == '__main__':
    main()
