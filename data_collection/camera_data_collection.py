#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import json
import os
import argparse

# Initialize the CvBridge class
bridge = CvBridge()

class StateUpdater:
    def __init__(self):
        self.cam_image = None
        self.cam_pose = {}
        self.point_pose = {}

    def cam_image_callback(self, msg):
        # Update the camera image
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.cam_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def cam_pose_callback(self, msg):
        # Update the camera pose
        position = msg.pose.position
        orientation = msg.pose.orientation
        self.cam_pose['position'] = [position.x, position.y, position.z]
        self.cam_pose['orientation'] = [orientation.x, orientation.y, orientation.z, orientation.w]

    def point_pose_callback(self, msg):
        # Update the point pose
        position = msg.pose.position
        self.point_pose['position'] = [position.x, position.y, position.z]

def main(args):
    rospy.init_node('data_listener')  # Replace 'your_node_name' with your desired node name

    # Create an instance of the StateUpdater class
    state_updater = StateUpdater()

    # Subscribe to the three topics and specify the callback functions
    rospy.Subscriber('/camera/color/image_raw', Image, state_updater.cam_image_callback)
    rospy.Subscriber(f'/natnet_ros/{args.rigid_body_label}/pose', PoseStamped, state_updater.cam_pose_callback)
    rospy.Subscriber('/natnet_ros/onemarker/pose', PoseStamped, state_updater.point_pose_callback)

    # Wait fsor the subscribers to initialize
    rospy.sleep(1)

    # create a clean directory to save the data
    output_dir = os.path.join(os.getcwd(), args.data_dir)
    if os.path.exists(output_dir):
        os.system('rm -rf ' + output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0 
    # Loop and save data when the 's' key is pressed
    while not rospy.is_shutdown():
        cv2.imshow('Camera Image', state_updater.cam_image)
        # Check if the 's' key is pressed
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            # make a copy of the current state
            current_state = state_updater.__dict__.copy()
            # Serialize the data to file
            out_path = os.path.join(output_dir, str(counter))
            os.makedirs(out_path)
            # save the image as png
            cv2.imwrite(os.path.join(out_path, 'image.png'), current_state['cam_image'])
            # save the poses in json 
            json_dict = {'camera': [current_state['cam_pose']['position'] + current_state['cam_pose']['orientation']],
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
    parser = argparse.ArgumentParser(description = "Camera data collection")
    parser.add_argument('-d', '--data_dir', type=str,
        default='camera_calib_data', help="directory to save the camera calibration data")
    parser.add_argument('-l', '--rigid_body_label', type=str,
        default='D405', help="rigid body's label in Motive tracking system. gelsightshare for Joe's setup and D405 for Ruihan's setup")
    args = parser.parse_args()
    main(args)

