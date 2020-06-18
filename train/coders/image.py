""" Encoders and decoders specific to tasks that operate over images. """

import torch
import torchvision.transforms as transforms
from coders.coder import Encoder, Decoder
import util.util


class ConcatenationEncoder(Encoder):
    """
    Concatenates `k` images into a single image. This class is currently only
    defined for `k = 2` and `k = 4`. For example, given `k = 2` 32 x 32
    (height x width) input images, this encoder downsamples each image to
    be 32 x 16 pixels in size, and then concatenate the two downsampled images
    side-by-side horizontally. Given `k = 4` 32 x 32 images, each image is
    downsampled to be 16 x 16 pixels in size and placed in quadrants of a
    resultant parity image.
    """

    def __init__(self, ec_k, ec_r, in_dim):
        super().__init__(ec_k, ec_r, in_dim)

        if ec_k != 2 and ec_k != 4:
            raise Exception(
            "ConcatenationEncoder currently supports values of `ec_k`of 2 or 4.")

        self.original_height = self.in_dim[2]
        self.original_width = self.in_dim[3]

        if (self.original_height % 2 != 0) or (self.original_width % 2 != 0):
            raise Exception(
                    "ConcatenationEncoder requires that image height and "
                    "width be divisible by 2. Image received with shape: "
                    + str(self.in_dim))

        if ec_k == 2:
            self.resized_height = self.original_height
            self.resized_width = self.original_width // 2
        else:
            # `ec_k` = 4
            self.resized_height = self.original_height // 2
            self.resized_width = self.original_width // 2

    def forward(self, in_data):
        batch_size = in_data.size(0)

        # Initialize a batch of parities to a tensor of all zeros
        out = util.util.try_cuda(
                torch.zeros(batch_size, 1,
                            self.original_height, self.original_width))

        reshaped = in_data.view(-1, self.num_in,
                                self.resized_height, self.resized_width)
        if self.num_in == 2:
            out[:, :, :, :self.resized_width] = reshaped[:, 0].unsqueeze(1)
            out[:, :, :, self.resized_width:] = reshaped[:, 1].unsqueeze(1)
        else:
            # `num_in` = 4
            out[:, :, :self.resized_height, :self.resized_width] = reshaped[:, 0].unsqueeze(1)
            out[:, :, :self.resized_height, self.resized_width:] = reshaped[:, 1].unsqueeze(1)
            out[:, :, self.resized_height:, :self.resized_width] = reshaped[:, 2].unsqueeze(1)
            out[:, :, self.resized_height:, self.resized_width:] = reshaped[:, 3].unsqueeze(1)

        return out

    def pre_tensor_transforms(self):
        """
        Returns
        -------
            List containing a single tranform that resizes images to be the
            size needed for concatenation.
        """
        return [transforms.Resize((self.resized_height, self.resized_width))]
