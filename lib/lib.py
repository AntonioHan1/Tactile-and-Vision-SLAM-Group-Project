import numpy as np
from scipy.spatial.transform import Rotation as R

def get_transform_matrix(tvec, quat):
  """
  Calculate the 4x4 homogeneous transformation matrix.
  :param tvec: np.1darray (3,); translation vector.
  :param quat: np.1darray (4,): quaternion vector.
  :return: np.2darray (4, 4); homogeneous transformation matrix.
  """
  T = np.eye(4)
  T[:3, :3] = R.from_quat(quat).as_matrix()
  T[:3, 3] = tvec
  return T

def camera_projection(points, mtx):
  """
  Get the pixel position of 3D points in the camera frame.
  :param points: np.2darray (N, 3); the points in camera frame.
  :param mtx: np.2darray (3, 3); the camera matrix.
  :return: np.2darray (N, 2); the pixel position of the points.
  """
  proj_points = (mtx @ points.T).T
  pxs = proj_points[:, :2] / proj_points[:, 2]
  return pxs
