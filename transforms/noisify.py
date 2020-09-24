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


class Mozaic(object):
    """Adds a mozaic effect to the given PIL Image.

    Args:
        area_ratio (float): ratio of the image area you want to mozaic (default: 0.10)
        min_width (int): Minimum value for width of mozaic area (default:40)
        max_width (int): Maximum value for width of mozaic area (default:100)
        min_strength (int): Minimum value for mozaic strength (default:3)
        max_strength (int): Maximum value for mozaic strength (default:10)
    """
    def __init__(self, area_ratio=0.10, min_width=40, max_width=100, min_strength=3, max_strength=10):
        self.area_ratio = area_ratio
        self.min_width = min_width
        self.max_width = max_width
        self.min_strength = min_strength
        self.max_strength = max_strength

    def _random_area(self, img_size):
        w,h = img_size
        area = w * h
        max_area = area * self.area_ratio #Default is 1/10th of image size
        dw = random.randint(self.min_width, self.max_width)
        dh = int(max_area//dw)
        #
        x = random.randint(0, w-dw)
        y = random.randint(0, h-dh)
        return (x,y,x+dw,y+dh)

    def _mozaic(self, img, box, strength=5):
        cpy = img.copy()
        cropped_img = cpy.crop(box)
        cw = cropped_img.width
        cy = cropped_img.height
        nx = max(4, cw//strength)
        ny = max(4, cy//strength)
        moz = cropped_img.resize((nx, ny), Image.NEAREST)
        moz = moz.resize((cw, cy), Image.NEAREST)
        cpy.paste(moz, box)
        return cpy

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be mozaiced

        Returns:
            PIL Image: mozaic image.
        """
        box = self._random_area(img.size)
        strength = random.randint(self.min_strength, self.max_strength)
        return self._mozaic(img, box, strength)

    def __repr__(self):
        return self.__class__.__name__ + f'(area_ratio={self.area_ratio}, width={self.min_width}-{self.max_width}, strength={self.min_strength}-{self.max_strength})'

