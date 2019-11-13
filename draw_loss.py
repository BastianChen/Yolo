import matplotlib.pyplot as plt
import numpy as np
import os

loss_path = "data/loss"
loss_txt_array = ["net_Adam_garbage_without_normal.txt", "net_Adam_garbage_with_normal.txt"]

# 残差块2层
new_loss = []
new_loss_array = []
# 残差块3层
old_loss = []
old_loss_array = []
for i, filename in enumerate(loss_txt_array):
    if i < 2:
        with open(os.path.join(loss_path, filename)) as file:
            loss_array = file.readlines()
            for loss in loss_array:
                if i == 0:
                    new_loss.append(np.array(loss.strip().split(), dtype=np.float))
                else:
                    old_loss.append(np.array(loss.strip().split(), dtype=np.float))
        if i == 0:
            new_loss = np.stack(new_loss)
            for j in range(0, new_loss.shape[0], 2):
                new_loss_array.append(new_loss[j:j + 2, 2].mean())
            new_loss_array = np.stack(new_loss_array)
        else:
            old_loss = np.stack(old_loss)
            for j in range(0, old_loss.shape[0], 2):
                old_loss_array.append(old_loss[j:j + 2, 2].mean())
            old_loss_array = np.stack(old_loss_array)

# 画出两种网络的损失图
plt.plot(new_loss_array)
plt.plot(old_loss_array)
plt.legend(['with', 'without'], loc='upper right')
plt.show()
