""" Encoder using a convolutional neural network """

import torch
from torch import nn

from coders.coder import Encoder


class ConvEncoder(Encoder):
    def __init__(self, ec_k, ec_r, in_dim, intermediate_channels_multiplier=20):
        """
        Parameters
        ----------
            intermediate_channels_multiplier: int
                Determines how many intermediate channels will be used in
                convolutions. The exact number of channels is determined by:
                `intermediate_channels_multiplier * ec_k`.
        """
        super().__init__(ec_k, ec_r, in_dim)
        self.ec_k = ec_k
        self.ec_r = ec_r

        self.act = nn.ReLU()
        int_channels = intermediate_channels_multiplier * ec_k

        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=self.ec_k, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=self.ec_r,
                      kernel_size=1, stride=1, padding=0, dilation=1)
        )

    def forward(self, in_data):
        val = in_data.view(-1, self.num_in,
                           self.in_dim[2], self.in_dim[3])

        out = self.nn(val)
        out = out.view(val.size(0), self.num_out, -1)
        return out
