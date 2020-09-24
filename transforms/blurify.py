#Blurify
import random
from PIL import Image
import cv2
import numpy as np

class MotionBlur(object):
    """Adds a motion blur effect to the given PIL Image.

    Args:
        min_size (int): Minimum value for kernel size in pixels (default:1)
        max_size (int): Maximum value for kernel size in pixels (default:50)
        min_angle (int): Minimum value for angle direction (default:0)
        max_angle (int): Maximum value for angle direction (default:360)
    """
    def __init__(self, min_size=1, max_size=50, min_angle=0, max_angle=360):
        self.min_size = min_size
        self.max_size = max_size
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be mozaiced

        Returns:
            PIL Image: blurred image.
        """
        size = random.randint(self.min_size, self.max_size)
        angle = random.randint(self.min_angle, self.max_angle)

        cv_im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        k = np.zeros((size, size), dtype=np.float32)
        k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
        k = k * ( 1.0 / np.sum(k) )
        r = cv2.filter2D(cv_im, -1, k)

        ret_img = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        return Image.fromarray(ret_img)

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.min_size}-{self.max_size}, angle={self.min_angle}-{self.max_angle})'


