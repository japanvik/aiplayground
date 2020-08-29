import torch
import torch.nn as nn
from .network import Network
from .attention import SelfAttention

class ImageDiscriminator(Network):
    """ A 256 x 256 image discriminator using SpectralNormalization, SELU and an Attention block
        takes a tensor of size (b,in_channles,256,256) as input where b is the batch size and the
        in_channels refer to the color depth fo the image
    """
    def __init__(self, in_channels=3, use_attention=True):
        super(ImageDiscriminator, self).__init__()

        def conv(in_channels, out_channels, kernel_size, stride, padding):
            normalization = nn.utils.spectral_norm
            yield normalization(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding))
            yield nn.SELU()

        model = []
        # Input Layer
        model += [nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)]
        model += [nn.SELU()]
        # Convolution Layers
        model += [*conv(in_channels= 64, out_channels=128, kernel_size=4, stride=2, padding=1)]
        model += [*conv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)]
        model += [*conv(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)]
        # Output Layer
        model += [nn.Conv2d(in_channels=512, out_channels=1, kernel_size=16, stride=1, padding=0)]
        model += [nn.Sigmoid()]

        if use_attention:
            model.insert(4, SelfAttention(128))

        # Build the model
        self.model = nn.Sequential(*model)


    def forward(self, input):
        x = self.model(input)
        return x.squeeze()


class PatchGANDiscrimnator(Network):

    def __init__(self, in_channels=3):
        super(PatchGANDiscrimnator, self).__init__()

        def conv(in_channels, out_channels, kernel_size, stride, padding):
            normalization = nn.utils.spectral_norm
            yield normalization(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding))
            yield nn.SELU()

        model = []
        # Input Layer
        model += [nn.Conv2d(in_channels=in_channels*2, out_channels=64, kernel_size=4, stride=2, padding=1)]
        model += [nn.SELU()]
        # Convolution Layer
        model += [*conv(in_channels= 64, out_channels=128, kernel_size=4, stride=2, padding=1)]
        model += [*conv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)]
        model += [*conv(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)]
        # Pad
        model += [nn.ZeroPad2d((1, 0, 1, 0))]
        # Output layer
        model += [nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1, bias=False)]
        model += [nn.Sigmoid()]

        # Build the model
        self.model = nn.Sequential(*model)


    def forward(self, image_A, image_B):
        # Combine the 2 images in to 1
        img_input = torch.cat((image_A, image_B), 1)
        return self.model(img_input)

