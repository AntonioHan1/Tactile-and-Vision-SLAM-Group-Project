import argparse
import cv2
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tactilevision.lib.lib import get_transform_matrix
from scipy.spatial.transform import Rotation as R


def gelsight_optimization(args):
    calib_data_dir = args.calib_data_dir
    calib_result_dir = args.calib_result_dir
    if not os.path.isdir(calib_result_dir):
        os.makedirs(calib_result_dir)
    # Filter out non-data directories
    experiment_reldirs = []
    for experiment_reldir in os.listdir(calib_data_dir):
        experiment_dir = os.path.join(calib_data_dir, experiment_reldir)
        if not os.path.isfile(os.path.join(experiment_dir, "point_in_gelsight.npy")):
            continue
        else:
            experiment_reldirs.append(experiment_reldir)
    # Sort the directories
    experiment_args = np.argsort([int(reldir) for reldir in experiment_reldirs])
    experiment_reldirs = np.array(experiment_reldirs)[experiment_args]
    # Get the poses
    p_gs = []
    p_ms = []
    for experiment_reldir in experiment_reldirs:
        experiment_dir = os.path.join(calib_data_dir, experiment_reldir)
        px_g = np.load(os.path.join(experiment_dir, "point_in_gelsight.npy"))
        with open(os.path.join(experiment_dir, "poses.json"), "r") as f:
            data = json.load(f)
            camera_pose = np.array(data["gelsight"])[0]
            point_pose = np.array(data["point"])
            # calculate point in marker frame
            T_m2w = get_transform_matrix(camera_pose[:3], camera_pose[3:])
            T_w2m = np.linalg.inv(T_m2w)
            p_w = np.ones(4)
            p_w[:3] = point_pose
            p_m = T_w2m @ p_w
            p_ms.append(p_m[:3])
            # calculate point in gelsight frame
            pointcloud = np.load(os.path.join(experiment_dir, "new_pcl.npy")).reshape(
                240, 320, 3
            )
            p_g = pointcloud[px_g[0], px_g[1]]
            p_g[2] = p_g[2] + 3.2 * 1e-3
            p_gs.append(p_g)
    p_gs = np.array(p_gs)
    p_ms = np.array(p_ms)
    # Find the transformation
    A = p_gs - np.mean(p_gs, axis=0)
    B = p_ms - np.mean(p_ms, axis=0)
    r, _ = R.align_vectors(A, B)
    r = r.as_matrix()
    t = np.mean(p_gs - (r @ p_ms.T).T, axis=0)
    T_m2g = np.eye(4)
    T_m2g[:3, :3] = r
    T_m2g[:3, 3] = t
    save_path = os.path.join(calib_result_dir, "marker2gelsight_transform.npy")
    print(f"Save gelsight to marker transformation to {save_path}")
    np.save(save_path, T_m2g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="optimizing marker to gelsight transformation"
    )
    parser.add_argument(
        "-d",
        "--calib_data_dir",
        type=str,
        default="/Users/joehuang/LocalDocuments/16833/Project/tactilevision/gelsight_calib_data",
    )
    parser.add_argument(
        "-r",
        "--calib_result_dir",
        type=str,
        default="/Users/joehuang/LocalDocuments/16833/Project/tactilevision/calib_result",
    )
    args = parser.parse_args()
    gelsight_optimization(args)
