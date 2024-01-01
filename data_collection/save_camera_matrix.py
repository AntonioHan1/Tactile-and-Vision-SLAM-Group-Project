import numpy as np
import pyrealsense2 as rs
import os
from pathlib import Path

def print_camera_intrinsics(pipeline):
    try:
        # Create a context object
        pipeline.start()
        frames = pipeline.wait_for_frames()
        intrinsics = frames[0].profile.as_video_stream_profile().intrinsics

        # Print camera intrinsics
        print(f"Camera intrinsics for RealSense camera:\n")
        print(f"Width: {intrinsics.width}")
        print(f"Height: {intrinsics.height}")
        print(f"Fx (focal length in x): {intrinsics.fx}")
        print(f"Fy (focal length in y): {intrinsics.fy}")
        print(f"Cx (principal point x): {intrinsics.ppx}")
        print(f"Cy (principal point y): {intrinsics.ppy}")

    finally:
        pipeline.stop()
        return intrinsics


def save_camera_intrinsics(connect_camera=False):
    if connect_camera:
        # connect to the camera and obtain its intrinsics
        
        # Initialize the RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Print camera intrinsics
        intrinsics = print_camera_intrinsics(pipeline)
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.ppx
        cy = intrinsics.ppy

    else:
        # use previous test results
        fx = 422.702392578125 
        fy = 422.702392578125 
        cx = 427.7725830078125 
        cy = 241.4703369140625


    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # save the camera matrix to npy file 
    current_file_path = Path(__file__).resolve()
    grandparent_path = current_file_path.parent.parent
    output_dir = os.path.join(str(grandparent_path), 'calib_result', 'camera_matrix.npy')
    print(f"Save camera matrix to {output_dir} \n{K}")
    np.save(output_dir, K)

if __name__ == '__main__':
    save_camera_intrinsics(connect_camera=False)
