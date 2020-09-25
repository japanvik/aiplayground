import torch.nn as nn
from .network import Network
from .attention import SelfAttention2D
from .resnet import ResNetBlock

class CycleGANGenerator(Network):
    """ Generator Network Following CycleGAN's Generator Architecture
        Takes 3 x 256 x 256 image tensor and outputs the same
    """

    def __init__(self, in_channels=3, out_channels=3, use_attention=True, residual_layers=6, use_dropout=True):
        super(CycleGANGenerator, self).__init__()

        def conv(in_channels, out_channels, kernel_size, stride, padding):
            yield nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding)
            yield nn.BatchNorm2d(out_channels)
            yield nn.LeakyReLU()

        model = []
        # Input Layer
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1)]
        model += [nn.LeakyReLU()]
        # Convolution
        model += [*conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)]
        if use_dropout:
            model += [nn.Dropout(p=0.2)]
        model += [*conv(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)]
        # ResNet
        model += [*[ResNetBlock(256)] * residual_layers]
        #Attention layer
        if use_attention:
            model += [SelfAttention2D(256)]
        # DeConvolution
        model += [nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)]
        model += [nn.LeakyReLU()]
        if use_dropout:
            model += [nn.Dropout(p=0.2)]
        model += [nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)]
        model += [nn.LeakyReLU()]
        #Output layer
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=7, stride=1)]
        model += [nn.Tanh()]

        # Build the model
        self.model = nn.Sequential(*model)


    def forward(self, input):
        x = self.model(input)
        return x

