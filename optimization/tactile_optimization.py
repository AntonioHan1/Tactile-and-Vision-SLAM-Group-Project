from __future__ import print_function

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np

# Most parts are directly edited based on a GTSAM example using a 3dof odometry[2].
# Here, tactile (3dof) odometry is used.
# Also, [1] and [3] are used as document/reference.

# noise, you can change the value
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.002, 0.002, 0.001]))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.003, 0.003, 0.001]))

def main():
    """Main runner"""
    
    # Load tactile odometry data. You can change the path or file name.
    tactile_pred = np.load('pred_deltas.npy')
    
    # Find total number of odometries
    n = tactile_pred.shape[0]
    
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    
    # Prior
    priorMean = gtsam.Pose2(0.0, 0.0, 0.0)  # prior at origin
    graph.add(gtsam.PriorFactorPose2(1, priorMean, PRIOR_NOISE))

    # Add odometry factors using loop
    for i in range(n):
        # same odometry noise being used for all factors for convenience, like in [2].
          odometry = gtsam.Pose2(tactile_pred[i,0],tactile_pred[i,1],(tactile_pred[i,2]/180.0*np.pi))
          graph.add(gtsam.BetweenFactorPose2(i+1, i+2, odometry, ODOMETRY_NOISE))


    # Initialize. Origin is used here for convenience, like in [2].
    initial = gtsam.Values()
    for i in range(1, n+2):
          initial.insert(i, gtsam.Pose2(0.0, 0.0, 0.0))


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
    optimized = np.zeros((n+1,3))
    for i in range(1, n+2):
        one_pose = result.atPose2(i)
        optimized[i-1] = [one_pose.x(),one_pose.y(),one_pose.theta()]

    # Save the optimized poses. You can change name or path
    np.save('11_28_3DOF_optimized_poses_prediction_tactile.npy',optimized)
    for i in range(1, n+2):
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5,
                              marginals.marginalCovariance(i))

    # put unit to the labels
    plt.xlabel('X axis (mm)')
    plt.ylabel('Y axis (mm)')
    plt.axis('equal')
    
    # You can change the name of saved image.
    plt.savefig('11_28_3DOF_prediction_tactile.png')

if __name__ == "__main__":
    main()


# Reference:
# [1] Factor graphs and gtsam. https://gtsam.org/tutorials/intro.html, accessed by December 13, 2023
# [2] Georgia Tech Research Corporation. Odometryexample.py. https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/OdometryExample.py, accessed by December 13, 2023.
# [3] gtsam::rot3 class reference. https://gtsam.org/doxygen/4.0.0/a02759.html, accessed by December 13, 2023