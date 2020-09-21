# resnet.py
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)]
        model += [nn.LeakyReLU()]
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)]
        model += [nn.BatchNorm2d(in_channels)]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        # add the shortcut too
        return x + self.model(x)

