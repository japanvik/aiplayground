# network.py
import os
import torch
import torch.nn as nn

class Network(nn.Module):

    @classmethod
    def init_weights(cls, m):
        classname = m.__class__.__name__
        if classname in ['ConvTranspose2d', 'Conv2d']:
            #torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            torch.nn.init.xavier_normal_(m.weight.data, gain=1) #gain 1
        elif classname in ['BatchNorm2d', 'Linear']:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def save(self, identifier, iteration, path):
        filename = f'{iteration}_{identifier}_{self.__class__.__name__}.pth'
        path = os.path.join(path, filename)
        # CPU-fy the model for compatibility
        torch.save(self.cpu().state_dict(), path)
        # convert back to GPU model
        if torch.cuda.is_available():
            self.cuda()

    def load(self, identifier, iteration, path):
        filename = f'{iteration}_{identifier}_{self.__class__.__name__}.pth'
        path = os.path.join(path, filename)
        self.load_state_dict(torch.load(path))

