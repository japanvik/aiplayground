# Black and white dataset based on a color picture

from PIL import Image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class BlackAndWhite(ImageFolder):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (source target) source is BnW image, target is color
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        # Transforms
        image_transform = transforms.Compose([transforms.Resize((288,288)),
                transforms.RandomRotation(degrees=5.0),
                transforms.RandomCrop(256)])
        gray_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3)])
        normalize_transform = transforms.Compose([ transforms.ToTensor() ])

        # Resize, rotate and crop
        target = image_transform(sample)
        source = gray_transform(target)
        #
        target = normalize_transform(target)
        source = normalize_transform(source)

        return source, target


