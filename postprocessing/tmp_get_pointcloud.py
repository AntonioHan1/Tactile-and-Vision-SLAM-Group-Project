import argparse
import os
import cv2
import numpy as np
from gslib.gs3drecon import Reconstruction3D, depth_to_points
import matplotlib.pyplot as plt

def get_pointcloud(args):
    mmpp = 0.0634  # mini gel 18x24mm at 240x320
    mpp = mmpp / 1000.
    # Reconstruct 3D object
    nn = Reconstruction3D(240, 320)
    nn.load_nn(os.path.join(args.model_path))
    dm_zero, _, _ = nn.get_depthmap(np.zeros((240, 320, 3)), mask_markers=False)
    nn.dm_zero = dm_zero

    # Visualization tools
    #viz = Visualize3D(240, 320)

    # Background image
    background_path = os.path.join(args.parent_dir, "background.png")
    background_image = cv2.imread(background_path)

    for dir in os.listdir(args.parent_dir):
        try:
            int(dir)
        except:
            continue
        print(dir)
        path = os.path.join(args.parent_dir, dir, "image_crop.png")
        image = cv2.imread(path)
        diff_image = image.astype(np.float32) - background_image.astype(np.float32)
        dm, _, _ = nn.get_depthmap(diff_image, mask_markers=False)
        points = depth_to_points(dm)
        points = np.reshape(points, (-1, 3)) * mpp
        save_path = os.path.join(args.parent_dir, dir, "new_pcl.npy")
        np.save(save_path, points)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="get pointcloud")
    parser.add_argument(
        "-p",
        "--parent_dir",
        type=str,
        default='/home/joehuang/16833_code/calib_data/gelsight_calib_data',
        help="parent directory of the data"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default='/home/joehuang/16833_code/data/ball_indenters/model/nnmini.pth',
        help="place where the model is saved"
    )
    args = parser.parse_args()
    get_pointcloud(args)
