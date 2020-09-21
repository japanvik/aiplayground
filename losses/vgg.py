import torch
from models.vgg19 import Vgg19Perceptual


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


