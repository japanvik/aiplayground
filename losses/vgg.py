import torch
from torch.nn import functional as F
from models.vgg19 import Vgg19Perceptual, Vgg19


class VGGLoss(torch.nn.Module):
    def __init__(self, weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19Perceptual()
        self.vgg.eval()
        self.vgg.cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights


    def get_gram_matrix(self, img):
        """
        Compute the gram matrix by converting to 2D tensor and doing dot product
        img: (batch, channel, height, width)
        """
        b, c, h, w = img.size()
        img = img.view(b*c, h*w)
        gram = torch.mm(img, img.t())
        return gram


    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        per_loss = 0
        style_loss = 0
        for i in range(len(x_vgg)):
            per_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            #target_gram_matrix = self.get_gram_matrix(target_feature)
        return per_loss


class PerceptualLoss(torch.nn.Module):
    """ Average MSE loss of each gram matrix of features, weighted by predefined weights
    """
    def __init__(self, weights=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg19()
        # Define the layer level weights
        if weights:
            self.weigths = weights
        else:
            self.weights = {'conv1_1': 0.75,
                         'conv2_1': 0.5,
                         'conv3_1': 0.2,
                         'conv4_1': 0.2,
                         'conv5_1': 0.2}

    def forward(self, source, target):
        source_layers = self.vgg(source)
        target_layers = self.vgg(target)
        loss = 0
        for layer in self.weights:
            input_feature = source_layers[layer]
            b, d, h, w = input_feature.shape
            layer_loss = self.weights[layer] * F.mse_loss(source_layers[layer], target_layers[layer])
            loss += layer_loss #/ (b * d * h * w)
        return loss

