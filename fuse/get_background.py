import cv2
import numpy as np
from utils import resize_crop_mini


data_path = '/home/joehuang/16833_code/6dof_data/tactile_and_pose.npz'
save_path = '/home/joehuang/16833_code/6dof_data/background.png'
keep_right_idx = 281
keep_left_idx = 349
data = np.load(data_path)
gs_images = data['gs_images']
gs_right_image = cv2.resize(gs_images[keep_right_idx], (3580, 2688))
gs_right_image = resize_crop_mini(gs_right_image, 320, 240)
gs_left_image = cv2.resize(gs_images[keep_left_idx], (3580, 2688))
gs_left_image = resize_crop_mini(gs_left_image, 320, 240)
gs_image = np.zeros_like(gs_right_image)
gs_image[:, :160, :] = gs_left_image[:, :160, :]
gs_image[:, 160:, :] = gs_right_image[:, 160:, :]

cv2.imwrite(save_path, gs_image)

cv2.imshow('gs_images', gs_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
