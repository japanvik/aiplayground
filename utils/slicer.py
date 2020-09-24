# Slicer - a utility to deal with batching images
#!pip install patchify

from PIL import Image
import math
import torch
import numpy as np
from patchify import patchify, unpatchify
from torchvision import transforms
from torchvision.transforms import functional as F


def tensor_to_pil(tensor):
    to_pil = transforms.ToPILImage()
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    return to_pil(image)

def stich(image_list, image_dim, layout_dim, crop_dim, kernel_size=256, overlap=40, debug=False):
    cvs = Image.new("RGB", image_dim, 'black')
    halflap = overlap//2
    cols, rows = layout_dim
    for r in range(rows):
        for c in range(cols):
            image_index = (cols*r)+c
            x_offset = 0 if c==0 else halflap
            y_offset = 0 if r==0 else overlap
            x = 0 if c==0 else (kernel_size-overlap) * c + halflap
            y = 0 if r==0 else (kernel_size-overlap) *r + overlap
            #256*2 - 40
            # paste the image
            cvs.paste(image_list[image_index].crop((x_offset, y_offset, kernel_size, kernel_size)), (x,y))
            if debug:
                #print(r,c,image_index, x_offset, x, y, kernel_size*r, overlap*(r-1))
                #if r>0: print(f"({kernel_size}-{halflap}) * {r} - ({halflap}*({r}-1))={y}")
                print(f"cvs.paste(image_list[{image_index}].crop(({x_offset}, {y_offset}, {kernel_size}, {kernel_size})), ({x},{y}))")
    return cvs.crop((0,0,crop_dim[0],crop_dim[1]))

def optimized_length(base, kernel_size, overlap):
    # return the optimal length, and the number of strides
    steps = math.ceil((base) / kernel_size)
    if kernel_size * steps - base - (overlap*(steps-1)) <= 0: steps+=1
    new_length = kernel_size * steps - (overlap * (steps-1))
    return new_length, steps


def recast_image(im, kernel_size=256, overlap=40):
    # return an image which is resized into a sliceable size
    im_w, im_h = im.size
    new_w, s_right = optimized_length(im_w, kernel_size, overlap)
    new_h, s_down = optimized_length(im_h, kernel_size, overlap)
    backdrop = Image.new("RGB", (new_w, new_h), 'black')
    backdrop.paste(im, (0,0))
    step = kernel_size-overlap
    patches = patchify(np.asarray(backdrop), (kernel_size, kernel_size,3), step=step) # split image into 256,256 3D patches
    return patches, (new_w, new_h), (s_right, s_down), backdrop

def batch(iterable, n=1):
    # Create batches from an iterable
    # n = number of elements per batch
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def slicer_render(im, model, kernel_size=256, overlap=40, batch_size=24):
    patches, image_dim, layout_dim, backdrop = recast_image(im, kernel_size=kernel_size, overlap=overlap)
    #print(patches.shape, image_dim, layout_dim)
    #
    full_length = layout_dim[0] * layout_dim[1]
    z = patches.reshape(full_length,kernel_size,kernel_size,3)
    #batch to max 24 images
    mini_batches = batch([F.to_tensor(x) for x in z], batch_size)
    # Run the rendering
    with torch.no_grad():
        image_list = []
        for b in mini_batches:
            output = model(torch.stack(b).cuda())
            image_list.extend([F.to_pil_image(x) for x in output.cpu()])
    final = stich(image_list, image_dim, layout_dim=layout_dim, kernel_size=kernel_size, overlap=overlap, crop_dim=im.size)
    return final
