import numpy as np
import os
import sys
import cv2
from gelsight import gsdevice
from gelsight import gs3drecon
import copy
import argparse

"""
Run the 3d reconstruction on the mini gelsight after data collection.
"""


def main(args):
    # Set flags
    MASK_MARKERS_FLAG = False
    GPU = False
    
    ''' Load neural network '''
    net_path = os.path.join(os.path.dirname(__file__), "gelsight/nnmini.pt")
    print('net path = ', net_path)

    VISUALIZE_DEPTH = True
    SAVE_DEPTH = True
    SAVE_PCL = True
    SAVE_CROP = True

    

    zeroing_num = args.zeroing_num # number of images used to zero the depth map

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320

    # This is meters per pixel that is used for ros visualization
    mpp = mmpp / 1000.

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gsdevice.Camera("GelSight Mini")

    if args.verbose: print(f"check dev dev imgw {dev.imgw} imgh {dev.imgh}")

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev, zeroing_num=zeroing_num)
    net = nn.load_nn(net_path, gpuorcpu)

    print('press q on image to exit')

    # ''' use this to plot just the 3d '''
    # vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)

    # Initialize point cloud
    x = np.arange(dev.imgw) * mpp
    y = np.arange(dev.imgh) * mpp
    X, Y = np.meshgrid(x, y)
    points = np.zeros([dev.imgw * dev.imgh, 3])
    points[:, 0] = np.ndarray.flatten(X)
    points[:, 1] = np.ndarray.flatten(Y)
    Z = np.zeros((dev.imgh, dev.imgw))  # initialize points array with zero depth values
    points[:, 2] = np.ndarray.flatten(Z)

    # sort the image folders so that the first few folders store the calibration GelSight images without any contact
    for img_folder in sorted(os.listdir(args.data_dir), key=lambda x: int(x)):
        img_path = os.path.join(args.data_dir, img_folder, 'image.png')
        f1 = cv2.imread(img_path)
        f1 = gsdevice.resize_crop_mini(f1, dev.imgw, dev.imgh)

        # compute the depth map
        dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)

        if args.verbose: print(f"dm shape {dm.shape} min {np.min(dm)} max {np.max(dm)}")

        # compute the point cloud
        dm_ros = copy.deepcopy(dm) * mpp 
        points[:, 2] = np.ndarray.flatten(dm_ros)

        # ''' Display the results '''
        # vis3d.update(dm)

        if VISUALIZE_DEPTH:
            # Display the image and wait for keypress
            cv2.imshow(f'depth map', dm)
            key = cv2.waitKey(0)
        
        if SAVE_CROP:
            crop_path = img_path.replace('image.png', 'image_crop.png')
            cv2.imwrite(crop_path, f1)

        if SAVE_DEPTH:
            output_path = img_path.replace('.png', '_depth.png')
            # min max normalization to 0-255 
            dm_normalize = (dm - np.min(dm)) / (np.max(dm) - np.min(dm)) * 255
            cv2.imwrite(output_path, dm_normalize)
        if SAVE_PCL:
            pcl_path = img_path.replace('image.png', 'pcl.npy')
            if args.verbose: print(f"Before saving pcl,points shape {points.shape}, sample points {points[0]} {points[1]}")
            np.save(pcl_path, points)
            if args.verbose: print(f'Saved point cloud to {pcl_path} pcl shape {points.shape}')

        # if 'q' is pressed then exit the loop
        if key & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Process GelSight depth map")
    parser.add_argument('-n', '--zeroing_num', type=int,
        default=50, help="number of images used to zero the depth map")
    parser.add_argument('-d', '--data_dir', type=str,
        default="/home/ruihan/Documents/Proj-3DTexture/exp/gelsight_calib_data", help="directory of raw GelSight image")
    parser.add_argument('-v', '--verbose', action='store_true', help="option to print out log messages")
    args = parser.parse_args()
    main(args)


