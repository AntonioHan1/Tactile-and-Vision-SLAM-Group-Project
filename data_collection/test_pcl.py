import numpy as np
import matplotlib.pyplot as plt

# Set the camera resolution
mmpp = 0.0634  # mini gel 18x24mm at 240x320

# This is meters per pixel that is used for ros visualization
mpp = mmpp / 1000.

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

data = np.load('/home/ruihan/Documents/Proj-3DTexture/exp/gelsight_calib_data/66/pcl.npy')

print(f"load data shape {data.shape}")
data = data.reshape(-1, 3)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.scatter(data[:,0]/mpp, data[:, 1]/mpp, data[:, 2]/mpp, c=data[:, 2])
ax.scatter(data[:,0], data[:, 1], data[:, 2], c=data[:, 2])
set_axes_equal(ax)
# label x, y, z
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
