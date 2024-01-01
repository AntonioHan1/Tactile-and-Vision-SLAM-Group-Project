import argparse
import cv2
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tactilevision.lib.lib import get_transform_matrix

def image_optimization(args):
  calib_data_dir = args.calib_data_dir
  calib_result_dir = args.calib_result_dir
  mtx = np.load(os.path.join(calib_result_dir, "camera_matrix.npy"))
  # Filter out non-data directories
  experiment_reldirs = []
  for experiment_reldir in os.listdir(calib_data_dir):
    experiment_dir = os.path.join(calib_data_dir, experiment_reldir)
    if not os.path.isfile(os.path.join(experiment_dir, "point_in_camera.npy")):
      continue
    else:
      experiment_reldirs.append(experiment_reldir)
  # Sort the directories
  experiment_args = np.argsort([int(reldir) for reldir in experiment_reldirs])
  experiment_reldirs = np.array(experiment_reldirs)[experiment_args]
  # Get the poses
  px_cs = []
  p_ms = []
  for experiment_reldir in experiment_reldirs:
    experiment_dir = os.path.join(calib_data_dir, experiment_reldir)
    px_c = np.load(os.path.join(experiment_dir, "point_in_camera.npy"))
    px_cs.append(px_c)
    with open(os.path.join(experiment_dir, 'poses.json'), 'r') as f:
      data = json.load(f)
      camera_pose = np.array(data["camera"])[0]
      point_pose = np.array(data["point"])
      T_m2w = get_transform_matrix(camera_pose[:3], camera_pose[3:])
      T_w2m = np.linalg.inv(T_m2w)
      p_w = np.ones(4)
      p_w[:3] = point_pose
      p_m = T_w2m @ p_w
      p_ms.append(p_m[:3])
  px_cs = np.array(px_cs, dtype=np.float32)
  p_ms = np.array(p_ms, dtype=np.float32)
  ret, rvecs, tvecs = cv2.solvePnP(p_ms, px_cs, mtx, np.zeros(5))
  # Construct the transformation matrix
  T_m2c = np.eye(4)
  T_m2c[:3, :3] = cv2.Rodrigues(rvecs)[0]
  T_m2c[:3, 3] = tvecs[:, 0]
  save_path = os.path.join(calib_result_dir, "marker2camera_transform.npy")
  print(f"Save camera to marker transformation to {save_path}")
  np.save(save_path, T_m2c)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "optimizing marker to camera transformation")
  parser.add_argument('-d', '--calib_data_dir', type=str,
      default="/Users/joehuang/LocalDocuments/16833/Project/tactilevision/data/cam_calib_data")
  parser.add_argument('-r', '--calib_result_dir', type=str,
       default="/Users/joehuang/LocalDocuments/16833/Project/tactilevision/data/calib_result")
  args = parser.parse_args()
  image_optimization(args)
