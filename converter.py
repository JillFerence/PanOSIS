"""
Code base from: https://github.com/nikhilroxtomar/RGB-Mask-to-Single-Channel-Mask-for-Multiclass-Segmentation/tree/main
"""

import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import re

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_mask(rgb_mask, colormap):
    output_mask = []

    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(rgb_mask, color), axis=-1)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)
    return output_mask


def get_unique_colors(mask):
    """Find unique colors in the RGB mask."""
    unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
    return unique_colors.tolist()

def generate_greyscale_colormap(n=100):
    # Generate grayscale shades from 0 to 255
    grayscale_values = np.linspace(0, 255, n).astype(int)
    return [[val, val, val] for val in grayscale_values]

def numerical_sort(file_list):
    return sorted(file_list, key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[0]))

if __name__ == "__main__":
    """ Create a directory """
    create_dir("OneShotMask-converted")

    """ Dataset paths """
    images = numerical_sort(glob(os.path.join("normal/images", "*.png")))
    masks = numerical_sort(glob(os.path.join("normal/mask", "*.png")))

    print(f"Images: {len(images)}")
    print(f"RGB Masks: {len(masks)}")

    """ VOC 2012 dataset: colormap and class names """
    VOC_COLORMAP = [
        [0, 0, 0],  [255, 0, 0],  [0, 255, 0],  [255, 255, 0],
        [0, 0, 255],  [255, 0, 255],  [0, 255, 255],  [255, 255, 255]
    ]

    VOC_CLASSES = [
        0, 1, 2, 3, 4, 5, 6, 7
    ]

    """ Displaying the class name and its pixel value """
    for name, color in zip(VOC_CLASSES, VOC_COLORMAP):
        print(f"{name} - {color}")

    count = 0

    """ Loop over the images and masks """
    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image and mask """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        mask = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Resizing the image and mask """
        # image = cv2.resize(image, (320, 320))
        # mask = cv2.resize(mask, (320, 320))

        """ Processing the mask to one-hot mask """
        processed_mask = process_mask(mask, VOC_COLORMAP)

        """ Converting one-hot mask to single channel mask """
        
        grayscale_mask = np.argmax(processed_mask, axis=-1)
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)

        final_path = os.path.join("converted", (str(count) + ".png"))
        cv2.imwrite(final_path, grayscale_mask)
        count += 1
