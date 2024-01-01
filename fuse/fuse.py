import argparse
import os
import cv2
import numpy as np
from tactilevision.fuse.utils import resize_crop_mini
from tactilevision.fuse.generate_contact_mask import get_contact_mask
from gslib.gs3drecon import Reconstruction3D, depth_to_points
from gslib.gsviz import Visualize3D
import matplotlib.pyplot as plt

def fuse(args):
    mmpp = 0.0634  # mini gel 18x24mm at 240x320
    mpp = mmpp / 1000.
    # Reconstruct 3D object
    nn = Reconstruction3D(240, 320)
    nn.load_nn(os.path.join(args.model_path))
    dm_zero, _, _ = nn.get_depthmap(np.zeros((240, 320, 3)), mask_markers=False)
    nn.dm_zero = dm_zero

    # Visualization tools
    viz = Visualize3D(240, 320)

    # Background image
    background_path = os.path.join(args.parent_dir, "background.png")
    background_image = cv2.imread(background_path)

    # Tactile images and poses
    data_path = os.path.join(args.parent_dir, "tactile_and_pose.npz")
    data = np.load(data_path)
    gs_images = data['gs_images']
    obj_poses = data['obj_poses']
    idxs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    pointcloud = np.empty((0,3))
    idx = 0
    while idx < len(idxs):
        gs_image = gs_images[idxs[idx]]
        gs_image = cv2.resize(gs_image, (3580, 2688))
        gs_image = resize_crop_mini(gs_image, 320, 240)
        diff_image = gs_image.astype(np.float32) - background_image.astype(np.float32)
        dm, gx, gy = nn.get_depthmap(diff_image, mask_markers=False)
        points = depth_to_points(dm, mpp)

        viz.update(dm)

        # Get contact mask
        contact_mask = get_contact_mask(gs_image.astype(np.float32), background_image.astype(np.float32), thresh=0.12, debug=False)[:, :, 0]
        contact_mask = np.logical_and(contact_mask, dm < -4.0)

        # Convert the boolean mask to uint8
        contact_mask_uint8 = (contact_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(contact_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gs_image_with_contours = cv2.drawContours(gs_image, contours, -1, (255, 255, 255), 1)


        # Obtain the contact mask
        points = points[contact_mask]
        obj_pose = obj_poses[idx]
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = (np.linalg.inv(obj_pose) @ points.T).T[:,:3]
        pointcloud = np.vstack((pointcloud, points))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c='b', s=0.1)
        # ax.set_aspect('equal')
        # plt.show()
        # Display the image
        cv2.imshow('Image with Contours', gs_image_with_contours)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Fuse GelSight reconstructed results"
    )
    parser.add_argument(
        "-p",
        "--parent_dir",
        type=str,
        help="parent directory of the data"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="place where the model is saved"
    )
    args = parser.parse_args()
    # Fuse the GelSight reconstructed results
    fuse(args)