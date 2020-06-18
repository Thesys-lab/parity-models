""" Encoders and decoders that use multilayer perceptrons (MLPs) """

import torch
from torch import nn

from coders.coder import Encoder, Decoder
from util.util import get_flattened_dim


def create_mlp(ec_k, ec_r, inout_dim, layer_sizes_multiplier):
    """
    Parameters
    ----------
        ec_k: int
            Number of input units for a forward pass of the coder.
        ec_r: int
            Number of output units from a forward pass of the coder.
        inout_dim: int
            Dimensionality of input units for a forward pass of the coder.
        layer_sizes_multiplier: list
            List of multipliers to be applied ot `inout_dim` that represents
            the width of each layer in the MLP. For example, a list of [2, 3]
            and `inout_dim` of 16 would create two input layers with 32 and
            48 hidden units, respectively.

    Returns
    -------
        ``torch.nn.Module``
            Module containing MLP to be used for this coder.
    """
    nn_modules = nn.ModuleList()
    prev_size = ec_k * inout_dim
    for i, size in enumerate(layer_sizes_multiplier):
        my_size = inout_dim * size
        l = nn.Linear(prev_size, my_size)
        prev_size = my_size
        nn_modules.append(l)
        nn_modules.append(nn.ReLU())

    nn_modules.append(nn.Linear(prev_size, ec_r * inout_dim))
    return nn.Sequential(*nn_modules)


class MLPEncoder(Encoder):
    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(num_in, num_out, in_dim, layer_sizes_multiplier)
        self.inout_dim = get_flattened_dim(in_dim)
        layer_sizes_multiplier = [ec_k]
        self.nn = create_mlp(ec_k, ec_r, self.inout_dim, layer_sizes_multiplier)

    def forward(self, in_data):
        # Flatten inputs
        val = in_data.view(in_data.size(0), -1)
        out = self.nn(val)
        return out.view(out.size(0), self.num_out, self.inout_dim)


class MLPDecoder(Decoder):
    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(num_in, num_out, in_dim, layer_sizes_multiplier)
        self.inout_dim = get_flattened_dim(in_dim)
        layer_sizes_multiplier = [ec_k, ec_r]
        self.nn = create_mlp(ec_k, ec_r, self.inout_dim, layer_sizes_multiplier)

    def forward(self, in_data):
        # Flatten inputs
        val = in_data.view(in_data.size(0), -1)
        out = self.nn(val)
        return out.view(out.size(0), self.num_out, self.inout_dim)
