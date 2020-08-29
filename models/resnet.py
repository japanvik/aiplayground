# resnet.py
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        snorm = nn.utils.spectral_norm

        model = []
        model += [nn.ReflectionPad2d(1)]
        model += [snorm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1))]
        model += [nn.SELU()]
        model += [nn.ReflectionPad2d(1)]
        model += [snorm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1))]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        # add the shortcut too
        return x + self.model(x)

