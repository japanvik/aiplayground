import torch.nn as nn
from .network import Network
from .attention import SelfAttention
from .resnet import ResNetBlock

class Pix2PixGenerator(Network):
    def __init__(self, in_channels=3, use_attention=True, residual_layers=7):
        super(Pix2PixGenerator, self).__init__()

        def conv(in_channels, out_channels, kernel_size, stride, padding):
            snorm = nn.utils.spectral_norm
            yield snorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding))
            yield nn.SELU()

        model = []
        # Input Layer
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=1)]
        model += [nn.SELU()]
        # Convolution
        model += [*conv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)]
        model += [nn.Dropout(p=0.2)]
        model += [*conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)]
        # ResNet
        model += [*[ResNetBlock(128)] * residual_layers]
        #Attention layer
        if use_attention:
            model += [SelfAttention(128)]
        # DeConvolution
        model += [nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)]
        model += [nn.SELU()]
        model += [nn.Dropout(p=0.2)]
        model += [nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)]
        model += [nn.SELU()]
        #Output layer
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, stride=1)]
        model += [nn.Tanh()]

        # Build the model
        self.model = nn.Sequential(*model)


    def forward(self, input):
        x = self.model(input)
        return x

