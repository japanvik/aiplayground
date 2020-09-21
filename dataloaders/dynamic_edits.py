# Dynamic Edits
from PIL import Image
from torchvision.datasets import folder
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import glob
import os

class DynamicEdits(ImageFolder):
    """ Dataset which points to a folder, but allows fine grained control
        of the resulting target image using transforms
    """

    def __getitem__(self, index):
        path, klass = self.samples[index]
        try:
            sample = self.loader(path).convert('RGB')
        except ValueError:
            raise ValueError('Error loading %s' % (path))

        if self.transform:
            source = self.transform(sample.copy())
        else:
            source = sample.copy()

        if self.target_transform:
            target = self.target_transform(source.copy())
        else:
            target = source.copy()

        source = transforms.ToTensor()(source)
        target = transforms.ToTensor()(target)

        return target, source





