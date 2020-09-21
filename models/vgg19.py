import torch
from torchvision import models

class Vgg19Perceptual(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19Perceptual, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        # replace max pooling to avg pooling
        for i, layer in enumerate(self.vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # Turn off gradients
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        # Cuda it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg.to(device).eval()

    def forward(self, image, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',  ## content layer
                      '28': 'conv5_1'}
            features = {}
            x = image
            for name, layer in enumerate(self.vgg.features):
                x = layer(x)
                if str(name) in layers:
                    features[layers[str(name)]] = x

        return features
