import numpy as np
from bag_helper_image import *
from visual_odom_nets import *

import matplotlib.pyplot as plt
import matplotlib

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

matplotlib.rcParams.update({'font.size': 45})

marker2gelsight_transform = np.load('../data/calib_result/marker2camera_transform.npy')
bag_file = 'vision_and_tactile.bag'

gs_image_times, pose_deltas, rs_image_pairs, rs_ordered, pose_ordered, _ = process_rs_bag(bag_file, marker2gelsight_transform)
#print(pose_deltas)
model = VisualOdometryModel()
model.load_weights('../checkpoints_12_5/visual_odom/epoch60/visual_odom_checkpoint')

# gs_image_times = gs_image_times[:4]
# pose_deltas = pose_deltas[:3]
# gs_image_pairs = gs_image_pairs[:3]

odom_prediction = [np.zeros(6)]
pred_deltas = []

# for pair in gs_image_pairs:
#   pred_delta = tf.squeeze(model(tf.expand_dims(pair, 0), training=False), 0)
#   pred_deltas.append(pred_delta.numpy())
#   odom_prediction.append(pred_delta.numpy() + odom_prediction[-1])


# pred_deltas = np.array(pred_deltas)

# np.save('pred_deltas.npy', pred_deltas)
# np.save('pose_deltas.npy', pose_deltas)
# np.save('gs_image_times.npy', gs_image_times)


# np.save('pred_deltas.npy', pred_deltas)
# np.save('pose_deltas.npy', pose_deltas)
# np.save('gs_image_times.npy', gs_image_times)

pred_deltas = np.load('../3dof_12_5_train_result/new_pred_deltas_ordered.npy')
pose_deltas = np.load('../3dof_12_5_train_result/new_pose_deltas_ordered.npy')
# pred= np.load('../checkpoints_12_5/new_pred_poses_ordered.npy')
# pose = np.load('../checkpoints_12_5/new_gt_pose_ordered.npy')

#gs_image_times = np.load('../checkpoints_12_5/new_gs_image_times.npy')

# gs_image_times = gs_image_times[0:len(pose_deltas)]
pred_deltas = pred_deltas[0:len(pose_deltas)]
# odom_prediction = np.array(odom_prediction)
# odom_actual = np.concatenate((np.zeros((1, 3)), np.cumsum(pose_deltas, axis=0)), axis=0)
# print(np.mean(pred_deltas[:, 0] -  pose_deltas[:, 0]), 
# np.mean(pred_deltas[:, 1] -  pose_deltas[:, 1]), 
# np.mean(pred_deltas[:, 2] -  pose_deltas[:, 2]), 
# np.mean(pred_deltas[:, 3] -  pose_deltas[:, 3]), 
# np.mean(pred_deltas[:, 4] -  pose_deltas[:, 4]), 
# np.mean(pred_deltas[:, 5] -  pose_deltas[:, 5]))
# print(len(pred_deltas))
plt.figure(figsize=(30, 15))
#print(gs_image_times)
gs_image_times = np.linspace(0, 17.5, len(pred_deltas))
ax = plt.gca()
plt.plot(gs_image_times, pred_deltas[:, 0], linewidth=2, label='Prediction')
plt.plot(gs_image_times, pose_deltas[:, 0], linewidth=2, label='Actual', linestyle='dashed')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Visual Odometry Performance\n(X Positional Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
plt.legend(loc='lower left', fontsize=40)
plt.savefig('../3dof_12_5_train_result/odom_preddeltas_x_pos_ordered.png')
plt.close()
# print(pose.shape)
# pred = pred[0:len(pose), :]
# print(pred.shape)
# gs_image_times = np.linspace(0, 17.5, len(pred))


# plt.figure(figsize=(30, 15))

# ax = plt.gca()
# plt.plot(gs_image_times, pred[:, 0], linewidth=2, label='Prediction')
# plt.plot(gs_image_times, pose[:, 0], linewidth=2, label='Actual', linestyle='dashed')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Visual Odometry Performance\n(X Positional Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
# plt.legend(loc='lower left', fontsize=40)
# plt.savefig('../checkpoints_12_5/odom_pose_x_pos_ordered.png')
# plt.close()
# plt.figure(figsize=(30, 15))

# ax = plt.gca()
# plt.plot(gs_image_times, pred[:, 1], linewidth=2, label='Prediction')
# plt.plot(gs_image_times, pose[:, 1], linewidth=2, label='Actual', linestyle='dashed')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Visual Odometry Performance\n(X Rotational Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
# plt.legend(loc='lower left', fontsize=40)
# plt.savefig('../checkpoints_12_5/odom_pose_x_rot_ordered.png')
# plt.close()
# plt.figure(figsize=(30, 15))

# ax = plt.gca()
# plt.plot(gs_image_times, pred[:, 2], linewidth=2, label='Prediction')
# plt.plot(gs_image_times, pose[:, 2], linewidth=2, label='Actual', linestyle='dashed')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Visual Odometry Performance\n(Y Positional Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
# plt.legend(loc='lower left', fontsize=40)
# plt.savefig('../checkpoints_12_5/odom_pose_y_pos_ordered.png')
# plt.close()
# plt.figure(figsize=(30, 15))

# ax = plt.gca()
# plt.plot(gs_image_times, pred[:, 3], linewidth=2, label='Prediction')
# plt.plot(gs_image_times, pose[:, 3], linewidth=2, label='Actual', linestyle='dashed')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Visual Odometry Performance\n(Y Rotational Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
# plt.legend(loc='lower left', fontsize=40)
# plt.savefig('../checkpoints_12_5/odom_pose_y_rot_ordered.png')
# plt.close()
# plt.figure(figsize=(30, 15))

# ax = plt.gca()
# plt.plot(gs_image_times, pred[:, 4], linewidth=2, label='Prediction')
# plt.plot(gs_image_times, pose[:, 4], linewidth=2, label='Actual', linestyle='dashed')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Visual Odometry Performance\n(Z Positional Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
# plt.legend(loc='lower left', fontsize=40)
# plt.savefig('../checkpoints_12_5/odom_pose_z_pos_ordered.png')
# plt.close()
# plt.figure(figsize=(30, 15))

# ax = plt.gca()
# plt.plot(gs_image_times, pred[:, 5], linewidth=2, label='Prediction')
# plt.plot(gs_image_times, pose[:, 5], linewidth=2, label='Actual', linestyle='dashed')
# plt.xlabel('Time (s)', labelpad=75, x=0.4)
# plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
# ax = plt.gca()
# ax.set_title('Visual Odometry Performance\n(Z Rotational Displacement)', y=1.05)
# ax.xaxis.set_tick_params(pad=50)
# ax.yaxis.set_tick_params(pad=50)
# plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
# plt.legend(loc='lower left', fontsize=40)
# plt.savefig('../checkpoints_12_5/odom_pose_z_rot_ordered.png')
# plt.close()


plt.figure(figsize=(30, 15))
ax = plt.gca()
plt.plot(gs_image_times, pred_deltas[:, 1], linewidth=2, label='Prediction')
plt.plot(gs_image_times, pose_deltas[:, 1], linewidth=2, label='Actual', linestyle='dashed')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Visual Odometry Performance\n(Y Positional Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
plt.legend(loc='lower left', fontsize=40)
plt.savefig('../3dof_12_5_train_result/odom_preddeltas_y_pos_ordered.png')
plt.close()


plt.figure(figsize=(30, 15))
ax = plt.gca()
plt.plot(gs_image_times, pred_deltas[:, 2], linewidth=2, label='Prediction')
plt.plot(gs_image_times, pose_deltas[:, 2], linewidth=2, label='Actual', linestyle='dashed')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacement\n(mm)', labelpad=75, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Visual Odometry Performance\n(Z Positional Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
plt.legend(loc='lower left', fontsize=40)
plt.savefig('../3dof_12_5_train_result/odom_preddeltas_z_pos_ordered.png')
plt.close()

plt.figure(figsize=(30, 15))
ax = plt.gca()
plt.plot(gs_image_times, pred_deltas[:, 3], linewidth=2, label='Prediction')
plt.plot(gs_image_times, pose_deltas[:, 3], linewidth=2, label='Actual', linestyle='dashed')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacment\n' + r'($^\circ$)', labelpad=75, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Visual Odometry Performance\n(X rotational Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
plt.legend(loc='lower left', fontsize=40)
plt.savefig('../3dof_12_5_train_result/odom_preddeltas_x_rot_ordered.png')
plt.close()

plt.figure(figsize=(30, 15))
ax = plt.gca()
plt.plot(gs_image_times, pred_deltas[:, 4], linewidth=2, label='Prediction')
plt.plot(gs_image_times, pose_deltas[:, 4], linewidth=2, label='Actual', linestyle='dashed')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacment\n' + r'($^\circ$)', labelpad=75, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Visual Odometry Performance\n(Y rotational Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
plt.legend(loc='lower left', fontsize=40)
plt.savefig('../3dof_12_5_train_result/odom_preddeltas_y_rot_ordered.png')
plt.close()

plt.figure(figsize=(30, 15))
ax = plt.gca()
plt.plot(gs_image_times, pred_deltas[:, 5], linewidth=2, label='Prediction')
plt.plot(gs_image_times, pose_deltas[:, 5], linewidth=2, label='Actual', linestyle='dashed')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Displacment\n' + r'($^\circ$)', labelpad=75, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Visual Odometry Performance\n(Z rotational Displacement)', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.2, bottom=0.25, top=0.85, right=0.95)
plt.legend(loc='lower left', fontsize=40)
plt.savefig('../3dof_12_5_train_result/odom_preddeltas_z_rot_ordered.png')
plt.close()