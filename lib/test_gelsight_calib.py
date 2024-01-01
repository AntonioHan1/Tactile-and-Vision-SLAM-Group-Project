import argparse
import json
import os
import numpy as np
from tactilevision.lib.lib import get_transform_matrix, camera_projection


def test_gelsight_calib(args):
    calib_data_dir = args.calib_data_dir
    calib_result_dir = args.calib_result_dir
    T_m2g = np.load(os.path.join(calib_result_dir, "marker2gelsight_transform.npy"))
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
    # Calculate difference between true pixel position and calculated pixel position
    for experiment_reldir in experiment_reldirs:
        experiment_dir = os.path.join(calib_data_dir, experiment_reldir)
        px = np.load(os.path.join(experiment_dir, "point_in_gelsight.npy"))
        pointcloud = np.load(os.path.join(experiment_dir, "new_pcl.npy")).reshape(
            240, 320, 3
        )
        p_g = pointcloud[px[0], px[1]]
        p_g[2] = p_g[2] + 3.2 * 1e-3
        with open(os.path.join(experiment_dir, "poses.json"), "r") as f:
            data = json.load(f)
            camera_pose = np.array(data["gelsight"])[0]
            point_pose = np.array(data["point"])
            T_m2w = get_transform_matrix(camera_pose[:3], camera_pose[3:])
            T_w2m = np.linalg.inv(T_m2w)
            p_w = np.ones(4)
            p_w[:3] = point_pose
            predicted_p_g = T_m2g @ T_w2m @ p_w
            print(
                "case %s, error: %.2f mm"
                % (experiment_reldir, np.linalg.norm(predicted_p_g[:3] - p_g) * 1000.0)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the library")
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
    test_gelsight_calib(args)
