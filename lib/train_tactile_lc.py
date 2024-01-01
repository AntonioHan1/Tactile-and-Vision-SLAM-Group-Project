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

marker2gelsight_transform = np.load('../data/calib_result/marker2gelsight_transform.npy')
bag_file = '../bags/vision_and_tactile.bag'

num_pos = 100 # Same number the original code used
num_neg = num_pos
gs_image_times, obj_poses_wrt_gs, gs_images = get_images_and_poses(bag_file, marker2gelsight_transform)

max_pos_trans = 10
min_neg_trans = 30

# Generate positive and negative pairs of indices of images (if I generate the image pairs themselves here,
# python crashes)
print('Generating index pairs')
pos_pairs = []
neg_pairs = []
for idx1, pose1 in enumerate(obj_poses_wrt_gs):
  for idx2, pose2 in enumerate(obj_poses_wrt_gs):
    if np.linalg.norm(pose1[:2] - pose2[:2]) < max_pos_trans:
      pos_pairs.append((idx1, idx2))
    elif np.linalg.norm(pose1[:2] - pose2[:2]) > min_neg_trans:
      neg_pairs.append((idx1, idx2))

pos_pairs = np.array(pos_pairs)
neg_pairs = np.array(neg_pairs)

# Only use a random subset of the generated index pairs
print('Subsampling index pairs')
pos_pairs = pos_pairs[np.random.choice(len(pos_pairs), size=num_pos, replace=False)]
neg_pairs = neg_pairs[np.random.choice(len(neg_pairs), size=num_neg, replace=False)]

# Split the image pairs into train and test sets
print('Splitting index pairs into train and test sets')
num_pos_train = int(0.8*num_pos)
num_pos_test = num_pos - num_pos_train

num_neg_train = int(0.8*num_neg)
num_neg_test = num_neg - num_neg_train

pos_train_idx = np.random.choice(num_pos, size=num_pos_train, replace=False)
mask = np.zeros(num_pos, dtype=np.bool)
mask[pos_train_idx] = True
not_mask = np.logical_not(mask)
pos_pairs_train = pos_pairs[mask]
pos_pairs_test = pos_pairs[not_mask]

neg_train_idx = np.random.choice(num_neg, size=num_neg_train, replace=False)
mask = np.zeros(num_neg, dtype=np.bool)
mask[neg_train_idx] = True
not_mask = np.logical_not(mask)
neg_pairs_train = neg_pairs[mask]
neg_pairs_test = neg_pairs[not_mask]

idx1_train = np.concatenate((pos_pairs_train[:, 0], neg_pairs_train[:, 0]))
idx2_train = np.concatenate((pos_pairs_train[:, 1], neg_pairs_train[:, 1]))
y_train = np.concatenate((np.ones(num_pos_train), np.zeros(num_neg_train)), 0)
y_train = np.stack((y_train, \
                    np.concatenate((np.zeros(num_pos_train), np.ones(num_neg_train)), 0)), 1)

idx1_test = np.concatenate((pos_pairs_test[:, 0], neg_pairs_test[:, 0]))
idx2_test = np.concatenate((pos_pairs_test[:, 1], neg_pairs_test[:, 1]))
y_test = np.concatenate((np.ones(num_pos_test), np.zeros(num_neg_test)), 0)
y_test = np.stack((y_test, \
                    np.concatenate((np.zeros(num_pos_test), np.ones(num_neg_test)), 0)), 1)

train_ds = tf.data.Dataset.from_tensor_slices(
    (idx1_train, idx2_train, y_train)).shuffle(100).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((idx1_test, idx2_test, y_test)).batch(32)

model = TactileLoopClosureModel()

loss_object = tfk.losses.CategoricalCrossentropy()
optimizer = tfk.optimizers.Adam(learning_rate=1e-4)

train_loss = tfk.metrics.Mean(name='train_loss')

test_loss = tfk.metrics.Mean(name='test_loss')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)

EPOCHS = 40

train_losses = []
test_losses = []

for epoch in range(EPOCHS):
  bt = time.perf_counter()
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  test_loss.reset_states()

  save_path = '../checkpoints/tactile_lc/epoch' + str(epoch) + '/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  model.save_weights(save_path + '/tactile_lc_checkpoint')

  for iteration, (idx1, idx2, labels) in enumerate(train_ds):
    print('Epoch %d train iteration %d' %(epoch + 1, iteration + 1)) 
    image_pairs = np.concatenate((gs_images[idx1], gs_images[idx2]), -1)
    train_step(image_pairs, labels)

  for iteration, (idx1, idx2, test_labels) in enumerate(test_ds):
    print('Epoch %d test iteration %d' %(epoch + 1, iteration + 1)) 
    image_pairs = np.concatenate((gs_images[idx1], gs_images[idx2]), -1)
    test_step(image_pairs, test_labels)

  at = time.perf_counter()

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Test Loss: {test_loss.result()}, '
    f'Wall time: {at - bt}, '
  )

  train_losses.append(train_loss.result().numpy().item())
  test_losses.append(test_loss.result().numpy().item())
  np.save('../checkpoints/tactile_lc/train_losses.npy', train_losses)
  np.save('../checkpoints/tactile_lc/test_losses.npy', test_losses)

save_path = '../checkpoints/tactile_lc/epoch' + str(EPOCHS) + '/'
if not os.path.exists(save_path):
  os.makedirs(save_path)
model.save_weights(save_path + '/tactile_lc_checkpoint')
