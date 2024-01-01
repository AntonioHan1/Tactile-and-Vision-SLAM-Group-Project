# Author: Arpit Agarwal
# Email: arpit15945@gmail.com
from skimage.filters import median, gaussian
from skimage.morphology import opening, closing, disk
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_contact_mask(hdr, bg, thresh = 2.0, debug=True):
  rgb_no_bg = hdr - bg

  bg_gaussian_blurred = bg
  for ch in range(3):
    bg_gaussian_blurred[..., ch] = gaussian(bg[..., ch], sigma=7)

  # if debug:
  #   plt.imshow(rgb_no_bg_gaussian_blurred)
  #   plt.title("gaussian blurred Diff HDR image")
  #   plt.show()

  rgb_no_bg = rgb_no_bg/(bg_gaussian_blurred+1e-15)

  for i in range(rgb_no_bg.shape[2]):
    rgb_no_bg[...,i] = median(rgb_no_bg[...,i], disk(3))
    rgb_no_bg[...,i] = closing(rgb_no_bg[...,i], disk(3))
    rgb_no_bg[...,i] = closing(rgb_no_bg[...,i], disk(5))

  if debug:
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(rgb_no_bg)
    axs[0].set_title("Diff image")
    axs[1].imshow(1.0*(rgb_no_bg>thresh))
    axs[1].set_title("mask area")
    plt.show()

  red, green, blue = np.dsplit(rgb_no_bg, 3)
  # contact_mask = (np.abs(rgb_no_bg)/np.mean(bg)) > 1e-2
  contact_mask = (np.abs(red) > thresh) | (np.abs(green) > thresh) | (np.abs(blue) > thresh)

  contact_mask[..., 0] = median(contact_mask[..., 0], disk(3))

  # perform some morphological operations

  # plt.imshow(contact_mask*1.0)
  # plt.title("Contact Mask")
  # plt.show()

  return contact_mask

if __name__ == "__main__":
  test_image = cv2.imread("/home/joehuang/16833_code/6dof_data/test_image.png").astype(np.float32)
  background_image = cv2.imread("/home/joehuang/16833_code/6dof_data/background.png").astype(np.float32)
  contact_mask = get_contact_mask(test_image, background_image, thresh = 0.2, debug=True)
  print(np.where(contact_mask))