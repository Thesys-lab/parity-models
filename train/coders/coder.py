import torch
import torch.nn as nn


class Coder(nn.Module):
    """
    Base class for implementing encoders and decoders. All new encoders and
    decoders should derive from this class.
    """

    def __init__(self, num_in, num_out, in_dim):
        """
        Parameters
        ----------
            num_in: int
                Number of input units for a forward pass of the coder.
            num_out: int
                Number of output units from a forward pass of the coder.
            in_dim: in_dim
                Dimension of flattened inputs to the coder.
        """
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out

    def forward(self, in_data):
        """
        Parameters
        ----------
            in_data: ``torch.autograd.Variable``
                Input data for a forward pass of the coder.
        """
        pass


class Decoder(Coder):
    """
    Class for implementing decoders. All new decoders should derive from this
    class.
    """

    def __init__(self, num_in, num_out, in_dim):
        super().__init__(num_in, num_out, in_dim)

    def forward(self, in_data):
        pass

    def combine_labels(self, in_data):
        """
        Parameters
        ----------
            in_data: ``torch.autograd.Variable``
                Input labels that are to be combined together.

        Returns
        -------
            Combination over in_data that can be used directly for the label in
            calculating loss for a parity model.
        """
        pass
