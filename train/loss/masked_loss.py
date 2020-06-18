import torch
import torch.nn as nn
import util.util


class MaskedLoss(nn.Module):
    """
    Wrapper around a loss function that applies a given mask to loss.
    """

    def __init__(self, base_loss):
        """
        Parameters
        ----------
            base_loss: PyTorch loss function
                The loss function to calculate before applying mask.
        """
        super().__init__()
        self.base_loss = util.util.construct(base_loss, {"reduction": "none"})

    def forward(self, preds, targets, mask):
        """
        Calculates loss between ``preds`` and ``targets`` and applies ``mask``
        to the outputs. The outputs are averaged in returning loss.
        """
        num_attempt = mask.nonzero().size(0)
        loss = self.base_loss(preds, targets)

        if len(loss.size()) > 1:
            mask = mask.unsqueeze(1).expand_as(loss)

        loss = loss * mask
        return torch.sum(loss) / num_attempt
