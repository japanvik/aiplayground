# Aligned Data set based on filename, differentiated by Directories A & B
import torch
import glob
import os
from PIL import Image
import torchvision.transforms as transforms

class Aligned(torch.utils.data.Dataset):
    def __init__(self, root, size=256, source_transforms=None, target_transforms=None, ext=["*.png", "*.jpg"]):
        super(Aligned, self).__init__()
        self.root = root
        self.size = size
        self.source_transforms = source_transforms
        self.target_transforms = target_transforms
        self.samples = self._make_dataset(sub_dir='A')
        self.loader =  Image.open


    def _make_dataset(self, sub_dir, formats=['*.jpg','*.png']):
        images = []
        image_path = os.path.join(self.root, sub_dir)
        for f in formats:
            fnames = glob.glob(os.path.join(image_path, f))
            images.extend(fnames)
        sorted(images)
        return images


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (source target) source is image in Directory A, B
        """
        path = self.samples[index]
        source = self.loader(path)
        target = self.loader(path.replace('/A/', '/B/'))

        # Transforms
        image_transform = transforms.Compose([transforms.Resize((self.size, self.size))])
        normalize_transform = transforms.Compose([ transforms.ToTensor() ])

        source = image_transform(source)
        target = image_transform(target)

        if self.source_transforms:
            source = self.source_transforms(source)
        if self.target_transforms:
            target = self.target_transforms(target)
        #
        source = normalize_transform(source)
        target = normalize_transform(target)

        return source, target


