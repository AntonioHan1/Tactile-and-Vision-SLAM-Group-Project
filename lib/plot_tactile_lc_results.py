import rosbag
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import bisect

from bag_helper import *
from tactile_odom_nets import *

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.random.seed(0)

tf.random.set_seed(0)

'''
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
'''

load_saved = False
save_path = '../saved_models/tactile_lc_12_10_23_100_pos_100_neg/'

if load_saved:
  pred_loop_closure_steps = np.load(save_path + 'pred_loop_closure_steps.npy')
  actual_loop_closure_steps = np.load(save_path + 'actual_loop_closure_steps.npy')
  gs_image_times = np.load(save_path + 'gs_image_times.npy')
else:
  marker2gelsight_transform = np.load('../data/calib_result/marker2gelsight_transform.npy')
  bag_file = '../bags/vision_and_tactile.bag'

  gs_image_times, obj_poses_wrt_gs, gs_images = get_images_and_poses(bag_file, marker2gelsight_transform)

  # Processing the whole bag takes a while, so just process the first 30 time steps
  gs_image_times = gs_image_times[:30]
  obj_poses_wrt_gs = obj_poses_wrt_gs[:30]
  gs_images = gs_images[:30]

  # Last 30 time steps
  '''
  gs_image_times = gs_image_times[-30:]
  obj_poses_wrt_gs = obj_poses_wrt_gs[-30:]
  gs_images = gs_images[-30:]
  '''

  model = TactileLoopClosureModel()
  model.load_weights(save_path + '/epoch40/tactile_lc_checkpoint')

  # Each element of pred_loop_closure_steps corresponds to a time step in the data.
  # Each element contains a list of previous time steps in the data that is predicted to have loop closure
  # with the current step
  pred_loop_closure_steps = [[] for step in range(len(gs_image_times))]

  # Ground truth version of the above array
  actual_loop_closure_steps = [[] for step in range(len(gs_image_times))]

  max_pos_trans = 10
  min_neg_trans = 30

  for cur_step, (cur_pose, cur_image) in enumerate(zip(obj_poses_wrt_gs, gs_images)):
    for prev_step, (prev_pose, prev_image) in enumerate(zip(obj_poses_wrt_gs[:cur_step], gs_images[:cur_step])):
      pair = np.concatenate((prev_image, cur_image), -1)
      lc_prob = model(tf.expand_dims(pair, 0), training=False)
      lc_prob = lc_prob.numpy()[0, 0]
      if lc_prob > 0.99:
        pred_loop_closure_steps[cur_step].append(prev_step)
      if np.linalg.norm(prev_pose[:2] - cur_pose[:2]) < max_pos_trans:
        actual_loop_closure_steps[cur_step].append(prev_step)

  np.save(save_path + 'gs_image_times.npy', gs_image_times)
  np.save(save_path + 'pred_loop_closure_steps.npy', pred_loop_closure_steps)
  np.save(save_path + 'actual_loop_closure_steps.npy', actual_loop_closure_steps)

import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import UnivariateSpline
matplotlib.rcParams.update({'font.size': 45})

plt.figure(figsize=(20, 15))
ax = plt.gca()
ax.scatter(gs_image_times, np.zeros_like(gs_image_times), s=500, color='black')
duration = gs_image_times[-1] - gs_image_times[0]
for cur_step, (pred, actual) in enumerate(zip(pred_loop_closure_steps, actual_loop_closure_steps)):
  true_positive = list(set(pred).intersection(set(actual)))
  false_positive = list(set(pred).difference(set(actual)))
  false_negative = list(set(actual).difference(set(pred)))

  colors = ['deepskyblue', 'red', 'orange']
  prev_step_lists = [true_positive, false_positive, false_negative]

  for color, prev_step_list in zip(colors, prev_step_lists):
    for prev_step in prev_step_list:
      prev_time = gs_image_times[prev_step]
      cur_time = gs_image_times[cur_step]
      mid_time = (prev_time + cur_time)/2
      spl = UnivariateSpline([prev_time, mid_time, cur_time], [0, (cur_time - prev_time)/duration, 0], k=2)
      spline_times = np.linspace(prev_time, cur_time, 100)
      ax.plot(spline_times, spl(spline_times), linewidth='10', color=color)

labels = ['True positive', 'False positive', 'False negative']
colors = ['deepskyblue', 'red', 'orange']
for label, color in zip(labels, colors):
  ax.plot([], [], linewidth='10', color=color, label=label)
plt.legend(loc='upper right', fontsize=60)
plt.title('Loop Closure')

plt.xlabel('Time (s)')
ax.set_yticks([])

plt.savefig('../plots/tactile_lc.png')
