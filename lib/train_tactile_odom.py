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
bag_file = '../bags/yaw_ring_gelsight_2023-11-10-20-18-58.bag'

_, pose_deltas, gs_image_pairs = process_gs_bag(bag_file, marker2gelsight_transform)

num_train = int(0.8*len(gs_image_pairs))
num_test = len(gs_image_pairs) - num_train

train_idx = np.random.choice(len(gs_image_pairs), size=(num_train,), replace=False)
mask = np.zeros(len(gs_image_pairs), dtype=np.bool)
mask[train_idx] = True
not_mask = np.logical_not(mask)

x_train = gs_image_pairs[mask]
y_train = pose_deltas[mask]

x_test = gs_image_pairs[not_mask]
y_test = pose_deltas[not_mask]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(100).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model = TactileOdometryModel()

loss_object = tfk.losses.MeanSquaredError()
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

  save_path = '../checkpoints/tactile_odom/epoch' + str(epoch) + '/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  model.save_weights(save_path + '/tactile_odom_checkpoint')

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  at = time.perf_counter()

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Test Loss: {test_loss.result()}, '
    f'Wall time: {at - bt}, '
  )

  train_losses.append(train_loss.result().numpy().item())
  test_losses.append(test_loss.result().numpy().item())
  np.save('../checkpoints/tactile_odom/train_losses.npy', train_losses)
  np.save('../checkpoints/tactile_odom/test_losses.npy', test_losses)

save_path = '../checkpoints/tactile_odom/epoch' + str(EPOCHS) + '/'
if not os.path.exists(save_path):
  os.makedirs(save_path)
model.save_weights(save_path + '/tactile_odom_checkpoint')
