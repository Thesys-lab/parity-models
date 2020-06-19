""" Encoders and decoders that use multilayer perceptrons (MLPs) """

import torch
from torch import nn

from coders.coder import Encoder, Decoder
from util.util import get_flattened_dim


class MLPEncoder(Encoder):
    def __init__(self, ec_k, ec_r, in_dim):
        """
        Arguments
        ---------
            ec_k: int
                Parameter k to be used in coded computation
            ec_r: int
                Parameter r to be used in coded computation
            in_dim: list
                List of sizes of input as (batch, num_channels, height, width).
        """
        super().__init__(ec_k, ec_r, in_dim)

        # The MLP encoder flattens image inputs before encoding. This function
        # gets the size of such flattened inputs.
        self.inout_dim = get_flattened_dim(in_dim)

        # Set up the feed-forward neural network consisting of two linear
        # (fully-connected) layers and a ReLU activation function.
        self.nn = nn.Sequential(
            nn.Linear(in_features=ec_k * self.inout_dim,
                      out_features=ec_k * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=ec_k * self.inout_dim,
                      out_features=ec_r * self.inout_dim)
        )

    def forward(self, in_data):
        # Flatten inputs
        val = in_data.view(in_data.size(0), -1)

        # Perform inference over encoder model
        out = self.nn(val)

        # The MLP encoder operates over different channels of input images
        # independently. Reshape the output to to form `ec_r` output images.
        return out.view(out.size(0), self.ec_r, self.inout_dim)


class MLPDecoder(Decoder):
    def __init__(self, ec_k, ec_r, in_dim):
        """
        Arguments
        ---------
            ec_k: int
                Parameter k to be used in coded computation
            ec_r: int
                Parameter r to be used in coded computation
            in_dim: list
                List of sizes of input as (batch, num_channels, height, width).
        """
        super().__init__(ec_k, ec_r, in_dim)

        # The MLP decoder flattens image inputs before encoding. This function
        # gets the size of such flattened inputs.
        self.inout_dim = get_flattened_dim(in_dim)
        num_in = ec_k + ec_r
        num_out = ec_k

        # Set up the feed-forward neural network consisting of two linear
        # (fully-connected) layers and a ReLU activation function.
        self.nn = nn.Sequential(
            nn.Linear(in_features=num_in * self.inout_dim,
                      out_features=num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=num_in * self.inout_dim,
                      out_features=num_out * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=num_out * self.inout_dim,
                      out_features=num_out * self.inout_dim)
        )

    def forward(self, in_data):
        # Flatten inputs
        val = in_data.view(in_data.size(0), -1)

        # Perform inference over encoder model
        out = self.nn(val)

        # Reshape the output to to form `ec_k` output images.
        return out.view(out.size(0), self.ec_k, self.inout_dim)
