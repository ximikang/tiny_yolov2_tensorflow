import numpy as np
import tensorflow as tf

cell_size = 13
cell
predictions = np.random.randn(13,13,5,125)
print(predictions.shape)
tx = predictions[:, :, :, 0]
ty = predictions[:, :, :, 1]
tw = predictions[:, :, :, 2]
th = predictions[:, :, :, 3]
tc = predictions[:, :, :, 4]
#x_offset = np.transpose(np.reshape(np.array([np.arange(cell_size)])))
center_x = tx*32
center_y = tf.sigmoid(ty)

