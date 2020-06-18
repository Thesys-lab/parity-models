import torch
from torch import nn

from coders.coder import Encoder, Decoder


class AdditionEncoder(Encoder):
    """
    Adds inputs together.
    """

    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)

    def forward(self, in_data):
        return torch.sum(in_data, dim=1).view(in_data.size(0), self.num_out, -1)


class SubtractionDecoder(Decoder):
    """
    Subtracts available outputs from parity output.
    """

    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)

    def forward(self, in_data):
        # Subtract availables from parity. Each group in the second dimension
        # already has "unavailables" zeroed-out.
        out = in_data[:, -1] - torch.sum(in_data[:, :-1], dim=1)
        out = out.unsqueeze(1).repeat(1, self.num_in, 1)

        return out

    def combine_labels(self, in_data):
        return torch.sum(in_data, dim=1)
