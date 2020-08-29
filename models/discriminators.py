import torch.nn as nn
from .network import Network
from .attention import SelfAttention

class ImageDiscriminator(Network):
    """ A 256 x 256 image discriminator using SpectralNormalization, SELU and an Attention block
        takes a tensor of size (b,3,256,256) as input where b is the batch size
    """
    def __init__(self, in_channels=3, use_attention=False):
        super(ImageDiscriminator, self).__init__()

        model = []
        # Input Layer
        model += [nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)]
        model += [nn.SELU()]
        # Convolution Layers
        SpectralNorm = nn.utils.spectral_norm
        model += [SpectralNorm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1))]
        model += [nn.SELU()]
        model += [SpectralNorm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1))]
        model += [nn.SELU()]
        model += [SpectralNorm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1))]
        model += [nn.SELU()]
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
