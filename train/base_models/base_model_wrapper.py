import torch
import torch.nn as nn


class BaseModelWrapper(nn.Module):
    """
    Wrapper around a base_model that resizes inputs to be 
    ``base_model_input_size`` prior to performing a forward pass on the base
    model.
    """

    def __init__(self, base_model, base_model_input_size):
        super().__init__()
        self.base_model = base_model
        self.base_model_input_size = base_model_input_size

    def forward(self, in_data):
        """
        Resizes ``in_data`` to ``base_model_input_size`` and returns result
        of a forward pass using this resized data on the base model.
        """
        in_data = in_data.view(*self.base_model_input_size)
        out = self.base_model(in_data)
        return out
