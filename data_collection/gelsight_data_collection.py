#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import json
import os
import open3d as o3d
import sensor_msgs.point_cloud2 as pcl2
import numpy as np
from sensor_msgs.msg import Image
import argparse


# Initialize the CvBridge class
bridge = CvBridge()

class StateUpdater:
    def __init__(self):
        self.gs_data = None
        self.gs_img = None
        self.gs_pose = {}
        self.point_pose = {}

    # def gs_pcd_callback(self, msg):
    #     # Update the gelsight point cloud
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(pcl2.read_points_list(msg))
    
    #     # Point cloud processing here
    #     self.gs_data = np.asarray(pc.points)

    def gs_img_callback(self, msg):
        # Update the gelsight image
        self.gs_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def gs_pose_callback(self, msg):
        # Update the gelsight pose
        position = msg.pose.position
        orientation = msg.pose.orientation
        self.gs_pose['position'] = [position.x, position.y, position.z]
        self.gs_pose['orientation'] = [orientation.x, orientation.y, orientation.z, orientation.w]

    def point_pose_callback(self, msg):
        # Update the point pose
        position = msg.pose.position
        self.point_pose['position'] = [position.x, position.y, position.z]

def main(args):
    rospy.init_node('data_listener')  # Replace 'your_node_name' with your desired node name

    # Create an instance of the StateUpdater class
    state_updater = StateUpdater()

    # Subscribe to the three topics and specify the callback functions
    # rospy.Subscriber('/gsmini_pcd', PointCloud2, state_updater.gs_pcd_callback)
    rospy.Subscriber('/gsmini_rawimg_0', Image, state_updater.gs_img_callback)
    rospy.Subscriber(f'/natnet_ros/{args.rigid_body_label}/pose', PoseStamped, state_updater.gs_pose_callback)
    rospy.Subscriber('/natnet_ros/onemarker/pose', PoseStamped, state_updater.point_pose_callback)

    # Wait fsor the subscribers to initialize
    rospy.sleep(1)

    gs_imgh = 320  
    gs_imgw = 240
    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320
    # This is meters per pixel that is used for ros visualization
    mpp = mmpp / 1000.

    # create a clean directory to save the data
    output_dir = os.path.join(os.getcwd(), args.data_dir)
    if os.path.exists(output_dir):
        os.system('rm -rf ' + output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0 
    # Loop and save data when the 's' key is pressed
    while not rospy.is_shutdown():
        # visualize the gelsight image
        if state_updater.gs_img is not None:
            cv2.imshow('gelsight image', state_updater.gs_img)

        # Check if the 's' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # make a copy of the current state
            current_state = state_updater.__dict__.copy()
            # Serialize the data to file
            out_path = os.path.join(output_dir, str(counter))
            os.makedirs(out_path)

            # # save the point cloud as npy
            # np.save(os.path.join(out_path, 'pc.npy'), current_state['gs_data'].reshape(gs_imgh, gs_imgw, 3))
   
            # save the image as png
            cv2.imwrite(os.path.join(out_path, 'image.png'), current_state['gs_img'])
            # save the poses in json 
            json_dict = {'gelsight': [current_state['gs_pose']['position'] + current_state['gs_pose']['orientation']],
                          'point': current_state['point_pose']['position']}
            with open(os.path.join(out_path, 'poses.json'), 'w') as json_file:
                json.dump(json_dict, json_file, indent=4)

            print('Saved data to ' + out_path)
            counter += 1
        
        # Check if the 'q' key is pressed
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "GelSight data collection")
    parser.add_argument('-d', '--data_dir', type=str,
        default='gelsight_calib_data', help="directory to save the GelSight calibration data")
    parser.add_argument('-l', '--rigid_body_label', type=str,
        default='GelSight', help="rigid body's label in Motive tracking system, gelsightshare for Joe's setup and GelSight for Ruihan's setup")
    args = parser.parse_args()
    main(args)

