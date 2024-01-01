from __future__ import print_function

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np

# Most parts are directly edited based on a GTSAM example using a 3dof odometry[2].
# But this one uses 6dof (visual) odometry.
# Also, [1] and [3] are used as document/reference.

# noise, you can change the value
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(0.01*np.array([0.2, 0.2, 0.1, 0.2, 0.2, 0.1]))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(0.01*np.array([0.3, 0.3, 0.1, 0.3, 0.3, 0.1]))

def main():
    """Main runner"""
    
    # Load tactile odometry data. You can change the path or file name.
    vision_pred = np.load('new_pred_deltas_ordered.npy')
    
    # Find total number of odometries
    n = vision_pred.shape[0]
    
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    
    # Prior
    priorMean = gtsam.Pose3(gtsam.Rot3(),gtsam.Point3(0.0, 0.0, 0.0))  # prior at origin
    graph.add(gtsam.PriorFactorPose3(1, priorMean, PRIOR_NOISE))

    # Add odometry factors using loop
    for i in range(n):
        # Translation
          translation = gtsam.Point3(vision_pred[i,0],vision_pred[i,1],(vision_pred[i,2]))
          
        # You can either use Rodrigues or Ypr to deal with rotation [3]. They yield similar results.
          rotation = gtsam.Rot3.Rodrigues((vision_pred[i,3]/180*np.pi,vision_pred[i,4]/180*np.pi,vision_pred[i,5]/180*np.pi) )
        #   rotation = gtsam.Rot3.Ypr( vision_pred[i,5]/180*np.pi,vision_pred[i,4]/180*np.pi,vision_pred[i,3]/180*np.pi )
        
        # Combine rotation and translation
          odometry = gtsam.Pose3(rotation, translation)
        # same odometry noise being used for all factors for convenience, like in [2].
          graph.add(gtsam.BetweenFactorPose3(i+1, i+2, odometry, ODOMETRY_NOISE))


    # Initialize.
    initial = gtsam.Values()
    for i in range(1, n+2):
          initial.insert(i, gtsam.Pose3(gtsam.Rot3(),gtsam.Point3(0.0, 0.0, 0.0)) )


    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    # marginal covariances
    marginals = gtsam.Marginals(graph, result)
    for i in range(1, n+2):
        print("X{} covariance:\n{}\n".format(i,
                                             marginals.marginalCovariance(i)))
    optimized = np.zeros((n+1,6))
    for i in range(1, n+2):
        one_pose = result.atPose3(i)
        optimized[i-1,:] = [one_pose.x(),one_pose.y(),one_pose.z(),
                          one_pose.rotation().roll(),one_pose.rotation().pitch(),
                          one_pose.rotation().yaw()]
        
    # Save the optimized poses. You can change name or path
    np.save('new_3DOF_optimized_poses_prediction_Ypr.npy',optimized)
    for i in range(1, n+2):
        gtsam_plot.plot_pose3(0, result.atPose3(i), 0.5,
                              marginals.marginalCovariance(i))
        
    # put unit to the labels
    plt.xlabel('X axis (mm)')
    plt.ylabel('Y axis (mm)')
    plt.gca().set_zlabel('Z axis (mm)')
    plt.axis('equal')
    
    # You can change the name of saved image.    
    plt.savefig('new_3DOF_prediction_Ypr.png')

if __name__ == "__main__":
    main()
    
# Reference:
# [1] Factor graphs and gtsam. https://gtsam.org/tutorials/intro.html, accessed by December 13, 2023
# [2] Georgia Tech Research Corporation. Odometryexample.py. https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/OdometryExample.py, accessed by December 13, 2023.
# [3] gtsam::rot3 class reference. https://gtsam.org/doxygen/4.0.0/a02759.html, accessed by December 13, 2023