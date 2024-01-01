import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import bisect

'''
These are the helper functions for visual training
'''
gs_img_height = 240
gs_img_width = 320
gs_img_channels = 3
IMG_SHAPE = (224, 224, 3)
IMG_OFFSET = (50, 150)  # offset when cropping the rgb images

def pose_msg_to_H(msg):
  pose = np.eye(4)
  pose[0, 3] = msg.pose.position.x
  pose[1, 3] = msg.pose.position.y
  pose[2, 3] = msg.pose.position.z
  quat = np.array([msg.pose.orientation.x, \
                   msg.pose.orientation.y, \
                   msg.pose.orientation.z, \
                   msg.pose.orientation.w])
  pose[:3, :3] = R.from_quat(quat).as_matrix()
  return pose

# def warp_img(img, output_sz):
      
#         WARP_W = output_sz[0]
#         WARP_H = output_sz[1]

#         points1=np.float32([[0,0],[480,0],[0,848],[480,848]])
#         points2=np.float32([[0,0],[WARP_W,0],[0,WARP_H],[WARP_W,WARP_H]])
#         matrix=cv2.getPerspectiveTransform(points1,points2)
#         result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))
#         return result
def add_padding(input_img ,top, bottom, left, right, color=[0, 0, 0]):
    # Read the image

    
    # Add padding
    padded_image = cv2.copyMakeBorder(input_img ,top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # Save the padded image
    return padded_image

def process_rgb(img):
        assert len(img.shape) == 3 and img.shape[2] == 3
        if img.shape == IMG_SHAPE:
            return img
        cropped = img[
            IMG_OFFSET[0]:IMG_OFFSET[0]+IMG_SHAPE[0], 
            IMG_OFFSET[1]:IMG_OFFSET[1]+IMG_SHAPE[1], :]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        assert cropped.shape == IMG_SHAPE
        return cropped

def compute_pose_delta_SE3(pose1, pose2):
  # Change in position in gs xy plane
  delta_pos = (pose2[:3, 3] - pose1[:3, 3]) * 1000

  # Change in orientation about gs z axis
  delta_rot = R.from_matrix(pose1[:3, :3]@pose2[:3, :3].transpose()).as_rotvec()*180/np.pi
  #print(delta_rot)
  return np.append(delta_pos, delta_rot)

def generate_index_pairs(list_size, pair_count, min_distance):
    pairs = []
    while len(pairs) < pair_count:
        a = np.random.randint(0, list_size - 1)
        b = np.random.randint(0, list_size - 1)
        # Ensure indices are at least `min_distance` apart
        if abs(a - b) >= min_distance:
            pair = tuple(sorted([a, b]))
            pairs.append(pair)
    
    return pairs

def process_rs_bag(bag_file, marker2camera_transform):
  bridge = CvBridge()

  # Tactile image, pose of gelsight wrt world, and pose of object
  # wrt world
  rs_image_topic = '/camera/color/image_raw'
  gs_marker_pose_topic = '/natnet_ros/gelsightshare/pose'
  obj_pose_topic = '/natnet_ros/object/pose'

  rs_image_times = []
  rs_images = []

  gs_marker_pose_times = []
  gs_marker_poses = []

  obj_pose_times = []
  obj_poses = []
  # print(rs_image_topic)
  for topic, msg, t in rosbag.Bag(bag_file).read_messages(topics=[rs_image_topic, gs_marker_pose_topic, obj_pose_topic]):
    if topic == rs_image_topic:
      rs_image_times.append(t.to_sec())
      pic = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
      #print(pic.shape)
      pic = add_padding(pic, 184, 184, 0, 0, [255,255,255])
      pic = cv2.resize(pic, (224,224))
      # cv2.imshow('pic',pic)
      # cv2.waitKey(0)
      rs_images.append(pic)
    elif topic == gs_marker_pose_topic:
      gs_marker_pose_times.append(t.to_sec())
      gs_marker_poses.append(pose_msg_to_H(msg))
    elif topic == obj_pose_topic:
      obj_pose_times.append(t.to_sec())
      obj_poses.append(pose_msg_to_H(msg))

  # For each image, compute the object pose wrt the gelsight frame
  obj_poses_wrt_camera = []
  for t in rs_image_times:
    gs_marker_pose_idx = bisect.bisect_left(gs_marker_pose_times, t)
    obj_pose_idx = bisect.bisect_left(obj_pose_times, t)
    if obj_pose_idx >= len(obj_poses) or gs_marker_pose_idx >= len(gs_marker_poses):
      break
   # print(marker2camera_transform@np.linalg.inv(gs_marker_poses[gs_marker_pose_idx])@obj_poses[obj_pose_idx])
    # print('here')
    # print(obj_pose_idx)
    # print('there')
    # print(gs_marker_pose_idx)
    # print(len(gs_marker_poses))
    # print(len(obj_poses))
    obj_poses_wrt_camera.append(marker2camera_transform@np.linalg.inv(gs_marker_poses[gs_marker_pose_idx])@obj_poses[obj_pose_idx])

  obj_poses_wrt_camera = np.array(obj_poses_wrt_camera)
  np.save('gt_poses_3dof.npy', obj_poses_wrt_camera)
  len_obj_poses_wrt_camera = len(obj_poses_wrt_camera)
  image_pairs_idx = generate_index_pairs(len_obj_poses_wrt_camera, 800, 30)
  
  pose_deltas = []
  rs_image_pairs = []
  for i in range(len(image_pairs_idx)):
    pose_deltas.append(compute_pose_delta_SE3(obj_poses_wrt_camera[image_pairs_idx[i][0]], obj_poses_wrt_camera[image_pairs_idx[i][1]]))
    image1 = rs_images[image_pairs_idx[i][0]]
    image2 = rs_images[image_pairs_idx[i][1]]
    rs_image_pairs.append(np.concatenate((image1, image2), 2))
  pose_deltas = np.array(pose_deltas)
  rs_image_pairs = np.array(rs_image_pairs)
  pose_deltas_ordered = np.array([compute_pose_delta_SE3(pose1, pose2) for pose1, pose2 in zip(obj_poses_wrt_camera[:-1], obj_poses_wrt_camera[1:])])

  rs_image_pairs_ordered = np.array([np.concatenate((image1, image2), 2) for image1, image2 in zip(rs_images[:-1], rs_images[1:])]).astype(np.float32)
  pose_deltas = pose_deltas[:len(rs_image_pairs)]
  rs_image_pairs = rs_image_pairs[:len(pose_deltas)]
  # print(len(pose_deltas))
  # print(len(rs_image_pairs))
  return rs_image_times, pose_deltas, rs_image_pairs, rs_image_pairs_ordered, pose_deltas_ordered, obj_poses_wrt_camera

def get_images_and_poses(bag_file, marker2gelsight_transform):
  bridge = CvBridge()

  # Tactile image, pose of gelsight wrt world, and pose of object
  # wrt world
  gs_image_topic = '/camera/color/image_raw'
  gs_marker_pose_topic = '/natnet_ros/gelsightshare/pose'
  obj_pose_topic = '/natnet_ros/object/pose'

  gs_image_times = []
  gs_images = []

  gs_marker_pose_times = []
  gs_marker_poses = []

  obj_pose_times = []
  obj_poses = []

  for topic, msg, t in rosbag.Bag(bag_file).read_messages(topics=[gs_image_topic, gs_marker_pose_topic, obj_pose_topic]):
    if topic == gs_image_topic:
      gs_image_times.append(t.to_sec())
      pic = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
      pic = add_padding(pic, 184, 184, 0, 0, [255,255,255])
      pic = cv2.resize(pic, (224,224))
      # cv2.imshow('pic',pic)
      # cv2.waitKey(0)
      gs_images.append(pic)
      #gs_images.append(bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
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
    if obj_pose_idx >= len(obj_poses) or gs_marker_pose_idx >= len(gs_marker_poses):
      break
    obj_poses_wrt_gs.append(marker2gelsight_transform@np.linalg.inv(gs_marker_poses[gs_marker_pose_idx])@obj_poses[obj_pose_idx])

  obj_poses_wrt_gs = np.array(obj_poses_wrt_gs)

  # Convert to SE(2)
  obj_poses_wrt_gs = np.array([compute_pose_delta_SE3(np.eye(4), pose) for pose in obj_poses_wrt_gs])

  obj_poses_wrt_gs = obj_poses_wrt_gs[:len(gs_images)]
  gs_images = gs_images[:len(obj_poses_wrt_gs)]
  gs_image_times = gs_image_times[:len(gs_images)]

  return gs_image_times, obj_poses_wrt_gs, np.array(gs_images).astype(np.float32)