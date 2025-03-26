import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000


image1 = Image.open(os.path.join(currentdir, "tmp", "00000_feat_x1.png"))
image2 = Image.open(os.path.join(currentdir, "tmp", "00000_feat_x2.png"))



def show_diff(image1, image2, name, threshold=0.01):
    image1 = np.array(image1).astype(np.float32)
    image2 = np.array(image2).astype(np.float32)
    diff1to2 = image1 - image2
    diff2to1 = image2 - image1
    # diff1to2 = np.clip(diff1to2, 0, 255)
    # diff2to1 = np.clip(diff2to1, 0, 255)
    diff1to2 = np.where(diff1to2 < threshold * 255, 0, 255)
    diff2to1 = np.where(diff2to1 < threshold * 255, 0, 255)
    row1 = np.concatenate([image1, image2], axis=1)
    row2 = np.concatenate([diff1to2, diff2to1], axis=1)
    diff = np.concatenate([row1, row2], axis=0)
    diff = Image.fromarray(diff.astype(np.uint8))
    diff.save(os.path.join(currentdir, "tmp", name + ".png"))

image1_up = image1.resize((image2.width, image2.height), Image.BICUBIC)
show_diff(image1_up, image2, name="diff1to2")