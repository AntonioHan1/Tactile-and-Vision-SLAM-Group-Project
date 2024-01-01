import tensorflow as tf
# import tensorflow.keras as tfk
# import tensorflow.keras.layers as tfkl

gs_img_channels = 3

'''
This is the training networks for visual part both odometry and loop closure
'''
class BranchModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.img_shape = (224,224, 3)

    self.model = tf.keras.applications.MobileNetV3Large(
            input_shape=self.img_shape,
            minimalistic=True,
            include_top=False,
            weights='imagenet',
            include_preprocessing=True)
    self.model.trainable = False

    self.flatten =  tf.keras.layers.Flatten()
    self.pooling = tf.keras.layers.GlobalAveragePooling2D()
    self.fc = tf.keras.layers.Dense(512, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)


  def call(self, x):

    x = self.model(x)
    x = self.pooling(x)
    return self.dropout(self.fc(self.flatten(x)))

class VisualOdometryModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    # Convolution branch for the first image in the pair
    self.pre_branch = BranchModel()
    self.post_branch = BranchModel()

    self.fc1 = tf.keras.layers.Dense(128, activation='relu')
    self.fc1_dropout = tf.keras.layers.Dropout(0.2)

    self.dense = tf.keras.layers.Dense(6, activation="linear", use_bias=True, 
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None)

  def call(self, x):
    #x = x/127.5 - 1
    pre_branch_res = self.pre_branch(x[:, :, :, :gs_img_channels])
    post_branch_res = self.post_branch(x[:, :, :, gs_img_channels:])
    joint_res = tf.keras.layers.concatenate([pre_branch_res, post_branch_res], axis=-1)
    return self.dense(self.fc1_dropout(self.fc1(joint_res)))
  
class VisualLoopClosureModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    # Convolution branch for the first image in the pair
    self.pre_branch = BranchModel()

    # Convolution branch for the second image in the pair
    self.post_branch = BranchModel()

    self.fc1 = tf.keras.layers.Dense(128, activation='relu')
    self.fc1_dropout = tf.keras.layers.Dropout(0.2)

    self.dense = tf.keras.layers.Dense(2, activation='softmax', use_bias=False)

  def call(self, x):
    #x = x/127.5 - 1
    pre_branch_res = self.pre_branch(x[:, :, :, :gs_img_channels])
    post_branch_res = self.post_branch(x[:, :, :, gs_img_channels:])
    joint_res = tf.keras.layers.concatenate([pre_branch_res, post_branch_res], axis=-1)
    return self.dense(self.fc1_dropout(self.fc1(joint_res)))