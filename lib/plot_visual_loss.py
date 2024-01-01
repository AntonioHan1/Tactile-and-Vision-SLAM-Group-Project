import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 45})

train_losses = np.load('../checkpoints_12_5_6dof/visual_odom/train_losses.npy')
test_losses = np.load('../checkpoints_12_5_6dof/visual_odom/test_losses.npy')
#print(test_losses)

print(train_losses.shape)
plt.figure(figsize=(30, 15))
ax = plt.gca()
plt.plot(train_losses, linewidth=10, label='Train')
plt.plot(test_losses, linewidth=10, label='Test')
plt.xlabel('Epoch', labelpad=75, x=0.4)
plt.ylabel('Loss', labelpad=100, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('3DoF Visual Odometry Training and Testing Loss Curves', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.15, bottom=0.25, top=0.85, right=0.85)
plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(1, 1))
plt.savefig('../checkpoints_12_5_6dof/loss_curves_12_5.png')
plt.close()