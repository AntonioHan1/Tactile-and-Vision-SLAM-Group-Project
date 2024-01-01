import numpy as np
from bag_helper import *
from tactile_odom_nets import *

import matplotlib.pyplot as plt
import matplotlib

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

matplotlib.rcParams.update({'font.size': 45})

marker2gelsight_transform = np.load('../data/calib_result/marker2gelsight_transform.npy')
bag_file = '../bags/yaw_ring_gelsight_2023-11-10-20-18-58.bag'

gs_image_times, pose_deltas, gs_image_pairs = process_gs_bag(bag_file, marker2gelsight_transform)

model = TactileOdometryModel()
model.load_weights('../saved_models/tactile_odom_11_27_23/epoch40/tactile_odom_checkpoint')

odom_prediction = [np.zeros(3)]

for pair in gs_image_pairs:
  pred_delta = tf.squeeze(model(tf.expand_dims(pair, 0), training=False), 0)
  odom_prediction.append(pred_delta.numpy() + odom_prediction[-1])

odom_prediction = np.array(odom_prediction)
odom_actual = np.concatenate((np.zeros((1, 3)), np.cumsum(pose_deltas, axis=0)), axis=0)

plt.figure(figsize=(20, 15))
ax = plt.gca()
plt.plot(gs_image_times, odom_prediction[:, 0], linewidth=10, label='Prediction')
plt.plot(gs_image_times, odom_actual[:, 0], linewidth=10, label='Actual')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacement\n(mm)', labelpad=175, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Tactile Odometry Performance\n(X Positional Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
plt.savefig('../plots/odom_prediction_x_pos.png')
plt.close()

plt.figure(figsize=(20, 15))
ax = plt.gca()
plt.plot(gs_image_times, odom_prediction[:, 1], linewidth=10, label='Prediction')
plt.plot(gs_image_times, odom_actual[:, 1], linewidth=10, label='Actual')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacement\n(mm)', labelpad=175, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Tactile Odometry Performance\n(Y Positional Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
plt.savefig('../plots/odom_prediction_y_pos.png')
plt.close()

plt.figure(figsize=(20, 15))
ax = plt.gca()
plt.plot(gs_image_times, odom_prediction[:, 2], linewidth=10, label='Prediction')
plt.plot(gs_image_times, odom_actual[:, 2], linewidth=10, label='Actual')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacment\n' + r'($^\circ$)', labelpad=175, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Tactile Odometry Performance\n(Z Rotational Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
plt.savefig('../plots/odom_prediction_z_rot.png')
plt.close()
