import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 45})

train_losses = np.load('../saved_models/tactile_odom_11_27_23/train_losses.npy')
test_losses = np.load('../saved_models/tactile_odom_11_27_23/test_losses.npy')

plt.figure(figsize=(20, 15))
ax = plt.gca()
plt.plot(train_losses, linewidth=10, label='Train')
plt.plot(test_losses, linewidth=10, label='Test')
plt.xlabel('Time (s)', labelpad=75, x=0.4)
plt.ylabel('Loss', labelpad=175, rotation=0, y=0.3)
ax = plt.gca()
ax.set_title('Loss Curves', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.35, bottom=0.25, top=0.85, right=0.6)
plt.legend(loc='upper right', fontsize=60, bbox_to_anchor=(2.6, 1))
plt.savefig('../plots/loss_curves.png')
plt.close()
