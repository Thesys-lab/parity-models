""" Encoders and decoders that use multilayer perceptrons (MLPs) """

import torch
from torch import nn

from coders.coder import Encoder, Decoder
from util.util import get_flattened_dim


class MLPEncoder(Encoder):
    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)
        self.inout_dim = get_flattened_dim(in_dim)
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
        out = self.nn(val)
        return out.view(out.size(0), self.num_out, self.inout_dim)


class MLPDecoder(Decoder):
    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)
        self.inout_dim = get_flattened_dim(in_dim)
        self.num_in = ec_k + ec_r
        self.num_out = ec_k
        self.nn = nn.Sequential(
            nn.Linear(in_features=self.num_in * self.inout_dim,
                      out_features=self.num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.num_in * self.inout_dim,
                      out_features=self.num_out * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.num_out * self.inout_dim,
                      out_features=self.num_out * self.inout_dim)
        )

    def forward(self, in_data):
        # Flatten inputs
        val = in_data.view(in_data.size(0), -1)
        out = self.nn(val)
        return out.view(out.size(0), self.num_out, self.inout_dim)
