import rosbag
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import bisect

gs_img_height = 240
gs_img_width = 320
gs_img_channels = 3


def pose_msg_to_H(msg):
    pose = np.eye(4)
    pose[0, 3] = msg.pose.position.x
    pose[1, 3] = msg.pose.position.y
    pose[2, 3] = msg.pose.position.z
    quat = np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
    )
    pose[:3, :3] = R.from_quat(quat).as_dcm()
    return pose


def compute_pose_delta_SE2(pose1, pose2):
    # Change in position in gs xy plane, in mm
    delta_pos = (pose2[:2, 3] - pose1[:2, 3]) * 1000

    # Change in orientation about gs z axis, in deg
    delta_rot = (
        R.from_dcm(np.dot(pose1[:3, :3], pose2[:3, :3].transpose())).as_rotvec()[2]
        * 180
        / np.pi
    )

    return np.append(delta_pos, delta_rot)


def save_tactile_and_pose(bag_file, save_file, marker2gelsight_transform):
    bridge = CvBridge()

    # Tactile image, pose of gelsight wrt world, and pose of object
    # wrt world
    gs_image_topic = "/gsmini_rawimg_0"
    gs_marker_pose_topic = "/natnet_ros/gelsightshare/pose"
    obj_pose_topic = "/natnet_ros/object/pose"

    gs_image_times = []
    gs_images = []

    gs_marker_pose_times = []
    gs_marker_poses = []

    obj_pose_times = []
    obj_poses = []

    for topic, msg, t in rosbag.Bag(bag_file).read_messages(
        topics=[gs_image_topic, gs_marker_pose_topic, obj_pose_topic]
    ):
        if topic == gs_image_topic:
            gs_image_times.append(t.to_sec())
            gs_images.append(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        elif topic == gs_marker_pose_topic:
            gs_marker_pose_times.append(t.to_sec())
            gs_marker_poses.append(pose_msg_to_H(msg))
        elif topic == obj_pose_topic:
            obj_pose_times.append(t.to_sec())
            obj_poses.append(pose_msg_to_H(msg))

    # For each image, compute the object pose wrt the gelsight frame
    obj_poses_wrt_gs = []
    for t in gs_image_times:
        gs_marker_pose_idx = bisect.bisect_left(gs_marker_pose_times, t)
        obj_pose_idx = bisect.bisect_left(obj_pose_times, t)
        obj_poses_wrt_gs.append(
            np.dot(
                marker2gelsight_transform,
                np.dot(
                    np.linalg.inv(gs_marker_poses[gs_marker_pose_idx - 1]),
                    obj_poses[obj_pose_idx - 1],
                ),
            )
        )
    gs_images = np.array(gs_images)
    obj_poses = np.array(obj_poses_wrt_gs)
    np.savez(save_file, gs_images=gs_images, obj_poses=obj_poses)


#   obj_poses_wrt_gs = np.array(obj_poses_wrt_gs)

#   pose_deltas = np.array([compute_pose_delta_SE2(pose1, pose2) for pose1, pose2 in zip(obj_poses_wrt_gs[:-1], obj_poses_wrt_gs[1:])])

#   gs_image_pairs = np.array([np.concatenate((image1, image2), 2) for image1, image2 in zip(gs_images[:-1], gs_images[1:])]).astype(np.float32)

#   return gs_image_times, pose_deltas, gs_image_pairs

marker2gelsight_transform = np.load(
    "/home/joehuang/16833_code/calib_data/calib_result/marker2gelsight_transform.npy"
)
bag_file = "/home/joehuang/16833_code/6dof_data/vision_and_tactile_6dof.bag"
save_file = "/home/joehuang/16833_code/6dof_data/tactile_and_pose.npz"
save_tactile_and_pose(bag_file, save_file, marker2gelsight_transform)

# tactile_data = np.load('/home/joehuang/16833_code/3dof_data/11_28_23_pose_deltas_tactile.npy')
# print(len(tactile_data))
