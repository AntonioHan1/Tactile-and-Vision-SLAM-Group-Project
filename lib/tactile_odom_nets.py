import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

gs_img_channels = 3

class BranchModel(tfk.Model):
  def __init__(self):
    super().__init__()
    self.conv1 = tfkl.Conv2D(32, 5, activation='relu')
    self.conv2 = tfkl.Conv2D(32, 5, activation='relu')
    self.pool1 = tfkl.MaxPooling2D(pool_size=(2,2), strides=2)

    self.conv3 = tfkl.Conv2D(32, 5, activation='relu')
    self.conv4 = tfkl.Conv2D(32, 5, activation='relu')
    self.pool2 = tfkl.MaxPooling2D(pool_size=(2,2), strides=2)

    self.conv5 = tfkl.Conv2D(32, 5, activation='relu')
    self.conv6 = tfkl.Conv2D(32, 5, activation='relu')
    self.pool3 = tfkl.MaxPooling2D(pool_size=(2,2), strides=2)

    self.flatten = tfkl.Flatten()
    self.fc = tfkl.Dense(256, activation="relu")
    self.dropout = tfkl.Dropout(0.2)

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.pool1(x)

    x = self.conv3(x)
    x = self.conv4(x)
    x = self.pool2(x)

    x = self.conv5(x)
    x = self.conv6(x)
    x = self.pool3(x)

    return self.dropout(self.fc(self.flatten(x)))

class TactileOdometryModel(tfk.Model):
  def __init__(self):
    super().__init__()
    # Convolution branch for the first image in the pair
    self.pre_branch = BranchModel()

    # Convolution branch for the second image in the pair
    self.post_branch = BranchModel()

    self.fc1 = tfkl.Dense(128, activation='relu')
    self.fc1_dropout = tfkl.Dropout(0.2)

    self.dense = tfkl.Dense(3)

  def call(self, x):
    x = x/127.5 - 1
    pre_branch_res = self.pre_branch(x[:, :, :, :gs_img_channels])
    post_branch_res = self.post_branch(x[:, :, :, gs_img_channels:])
    joint_res = tfkl.concatenate([pre_branch_res, post_branch_res], axis=-1)
    return self.dense(self.fc1_dropout(self.fc1(joint_res)))

class TactileLoopClosureModel(tfk.Model):
  def __init__(self):
    super().__init__()
    # Convolution branch for the first image in the pair
    self.pre_branch = BranchModel()

    # Convolution branch for the second image in the pair
    self.post_branch = BranchModel()

    self.fc1 = tfkl.Dense(128, activation='relu')
    self.fc1_dropout = tfkl.Dropout(0.2)

    self.dense = tfkl.Dense(2, activation='softmax', use_bias=False)

  def call(self, x):
    x = x/127.5 - 1
    pre_branch_res = self.pre_branch(x[:, :, :, :gs_img_channels])
    post_branch_res = self.post_branch(x[:, :, :, gs_img_channels:])
    joint_res = tfkl.concatenate([pre_branch_res, post_branch_res], axis=-1)
    return self.dense(self.fc1_dropout(self.fc1(joint_res)))
