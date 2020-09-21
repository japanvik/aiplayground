#jpegify.py
import random
import io
from PIL import Image

class JpegCompress(object):
    """Adds Jpeg Compression to the given PIL Image.
    Picks a random compression value between the min and max quality arguments.

    Args:
        min_quality (int): Minimum value for jpeg compression
        max_quality (int): Maximum value for jpeg compression
    """

    def __init__(self, min_quality=5, max_quality=50):
        self.min_quality = min_quality
        self.max_quality = max_quality


    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be compressed.

        Returns:
            PIL Image: Compressed image.
        """
        quality = random.randint(self.min_quality, self.max_quality)
        output = io.BytesIO()
        img.save(output, format="jpeg", quality=quality)
        output.seek(0)
        r = Image.open(output)
        #output.close()
        return r


    def __repr__(self):
        return self.__class__.__name__ + f'(min_quality={self.min_quality}, max_quality={self.max_quality})'


