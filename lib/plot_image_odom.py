import numpy as np
from bag_helper_image import *
from visual_odom_nets import *

import matplotlib.pyplot as plt
import matplotlib

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

matplotlib.rcParams.update({'font.size': 45})

marker2gelsight_transform = np.load('../data/calib_result/marker2camera_transform.npy')
bag_file = 'vision_and_tactile_6dof.bag'

gs_image_times, pose_deltas, gs_image_pairs,rs_img_pairs_ordered, pose_deltas_ordered, _ = process_rs_bag(bag_file, marker2gelsight_transform)
#print(pose_deltas)
model = VisualOdometryModel()
model.load_weights('../checkpoints_12_5_6dof/visual_odom/epoch60/visual_odom_checkpoint')

# gs_image_times = gs_image_times[:4]
# pose_deltas = pose_deltas[:3]
# gs_image_pairs = gs_image_pairs[:3]

odom_prediction = [np.zeros(6)]
pred_deltas = []

for pair in rs_img_pairs_ordered:
  pred_delta = tf.squeeze(model(tf.expand_dims(pair, 0), training=False), 0)
  pred_deltas.append(pred_delta.numpy())
  odom_prediction.append(pred_delta.numpy() + odom_prediction[-1])

odom_prediction = np.array(odom_prediction)

odom_actual = np.concatenate((np.zeros((1, 6)), np.cumsum(pose_deltas_ordered, axis=0)), axis=0)
pred_deltas = np.array(pred_deltas)

# np.save('../checkpoints_12_5/new_pred_deltas_ordered.npy', pred_deltas)
# np.save('../checkpoints_12_5_6dof/new_pose_deltas_ordered.npy', pose_deltas_ordered)
np.save('../checkpoints_12_5_6dof/new_pred_poses_ordered.npy', odom_prediction)
np.save('../checkpoints_12_5_6dof/new_gt_pose_ordered.npy', odom_actual)


#np.save('../checkpoints_12_5_6dof/new_gs_image_times.npy', gs_image_times)

# odom_prediction = np.array(odom_prediction)

# odom_actual = np.concatenate((np.zeros((1, 6)), np.cumsum(pose_deltas, axis=0)), axis=0)

# plt.figure(figsize=(20, 15))
# ax = plt.gca()
# plt.plot(gs_image_times, odom_prediction[:, 0], linewidth=10, label='Prediction')
# plt.plot(gs_image_times, odom_actual[:, 0], linewidth=10, label='Actual')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=175, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Tactile Odometry Performance\n(X Positional Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
# plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
# plt.savefig('../plots_vnt/odom_prediction_x_pos.png')
# plt.close()

# plt.figure(figsize=(20, 15))
# ax = plt.gca()
# plt.plot(gs_image_times, odom_prediction[:, 1], linewidth=10, label='Prediction')
# plt.plot(gs_image_times, odom_actual[:, 1], linewidth=10, label='Actual')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=175, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Tactile Odometry Performance\n(Y Positional Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
# plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
# plt.savefig('../plots_vnt/odom_prediction_y_pos.png')
# plt.close()

# plt.figure(figsize=(20, 15))
# ax = plt.gca()
# plt.plot(gs_image_times, odom_prediction[:, 2], linewidth=10, label='Prediction')
# plt.plot(gs_image_times, odom_actual[:, 2], linewidth=10, label='Actual')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacment\n' + r'($^\circ$)', labelpad=175, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Tactile Odometry Performance\n(Z Rotational Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
# plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
# plt.savefig('../plots_vnt/odom_prediction_z_rot.png')
# plt.close()

# plt.figure(figsize=(20, 15))
# ax = plt.gca()
# plt.plot(gs_image_times[:-1], pred_deltas[:, 0], linewidth=10, label='Prediction')
# plt.plot(gs_image_times[:-1], pose_deltas[:, 0], linewidth=10, label='Actual')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=175, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Tactile Odometry Performance\n(X Positional Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
# plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
# plt.savefig('../plots_vnt/odom_preddeltas_x_pos.png')
# plt.close()

# plt.figure(figsize=(20, 15))
# ax = plt.gca()
# plt.plot(gs_image_times[:-1], pred_deltas[:, 1], linewidth=10, label='Prediction')
# plt.plot(gs_image_times[:-1], pose_deltas[:, 1], linewidth=10, label='Actual')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=175, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Tactile Odometry Performance\n(Y Positional Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
# plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
# plt.savefig('../plots_vnt/odom_preddeltas_y_pos.png')
# plt.close()