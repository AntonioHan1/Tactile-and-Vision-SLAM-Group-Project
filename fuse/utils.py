import cv2
import numpy as np

def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
        np.floor(img.shape[1] * (1 / 7))
    )
    # keep the ratio the same as the original image size
    img = img[
        border_size_x + 2 : img.shape[0] - border_size_x,
        border_size_y : img.shape[1] - border_size_y,
    ]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img
