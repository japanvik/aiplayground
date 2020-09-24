# image_utils
from PIL import Image


def scale(image, max_size, method=Image.BICUBIC, bg='black'):
    """
    resize 'image' to 'max_size' keeping the aspect ratio
    and place it in center of white 'max_size' image
    """
    image.thumbnail(max_size, method)
    offset = (int((max_size[0] - image.size[0]) / 2), int((max_size[1] - image.size[1]) / 2))
    back = Image.new("RGB", max_size, bg)
    back.paste(image, offset)

    return back

