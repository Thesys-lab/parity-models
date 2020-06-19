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
            in_dim: list
                List of sizes of input as (batch, num_channels, height, width).
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


class Encoder(Coder):
    """
    Class for implementing encoders. All new encoders should derive from this
    class.
    """

    def __init__(self, num_in, num_out, in_dim):
        super().__init__(num_in, num_out, in_dim)
        if len(in_dim) == 2:
            # Some `in_dim` values for square inputs with a single input
            # channel are represented using only a single value. We reconstruct
            # these into a (num_channels, height, width) here.
            flattened_dim = in_dim[1]
            if int(flattened_dim ** 0.5) ** 2 != flattened_dim:
                raise Exception(
                "Expected square flattened input, but received flattened "
                "input of size " + str(flattened_dim))
            sqrt_flattened = int(flattened_dim ** 0.5)
            self.in_dim = [-1, 1, sqrt_flattened, sqrt_flattened]
        else:
            self.in_dim = in_dim

    def forward(self, in_data):
        pass

    def resize_transform(self):
        """
        Returns
        -------
            A `torchvision.transforms.Transform` object that should be
            applied to data samples prior to being encoded. This transformation
            will be performed just before the data sample has been reformatted
            as a PyTorch `Tensor` object using `torchvision.transforms.ToTensor()`.
            This method only needs to be implemented if the encoder takes in
            images with a different size than those stored in the underlying
            dataset.
        """
        return None


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
