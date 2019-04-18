import sys, os, time
import math
import tensorflow as tf
import numpy as np
import cv2

input_height = 416
input_width = 416
bn_epsilon = 1e-3
n_input_imgs = 1
relu_alpha = 0.1

with tf.name_scope('input'):
    images = tf.placeholder(tf.float32, shape=[n_input_imgs,
                                          input_height,
                                          input_width,
                                          3])
    labels = tf.placeholder(tf.float32, shape=[n_input_imgs, 1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def max_pool_layer(input_tensor,kernel_size=2,stride=2,padding="VALID"):
    pooling_result = tf.nn.max_pool(input_tensor, ksize=[1,kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding)
    return pooling_result

def leaky_relu(x, alpha=relu_alpha):
    #return tf.nn.relu_layer(x, alpha=alpha)
    return tf.maximum(alpha * x, x)

def conv2(input_tensor, weight, strides=1,padding='SAME'):
    result = tf.nn.conv2d(input_tensor, weight, strides=[strides, strides, strides, strides],
                          padding=padding)
    return result

n_params = 0

#conv1 416*146*3 -> 416*416*16
w1 = weight_variable([3, 3, 3, 16])
b1 = bias_variable([16])
c1 = conv2(images, w1) + b1
conv1 = leaky_relu(c1)
n_params = n_params + 3*3*3*16 + 16*4

#max1 416*416*16 -> 208*208*16
max1 = max_pool_layer(conv1)

#conv3 208*208*16 -> 208*208*32
w2 = weight_variable([3, 3, 16, 32])
b2 = bias_variable([32])
print(n_params, max1, w2, b2)
c2 = conv2(max1, w2) + b2
conv2 = leaky_relu(c2)
n_params = n_params + 3*3*16*32 + 32*4

#max4 208*208*32 -> 104*104*32
max2 = max_pool_layer(conv2)

#conv5 104*104*32 -> 104*104*64
w3 = weight_variable([3, 3, 32, 64])
b3 = bias_variable([64])
print(n_params, max2, w3, b3)
c3 = conv2(max2, w3) + b3
conv3 = leaky_relu(c3)
n_params = n_params + 3*3*32*64 + 64*4

#max6 104*104*64 -> 52*52*64
max3 = max_pool_layer(conv3)

#conv7 52*52*64 -> 52*52*128
w4 = weight_variable([3, 3, 64, 128])
b4 = bias_variable([128])
c4 = conv2(max3, w4) + b4
conv4 = leaky_relu(c4)
n_params = n_params + 3*3*64*128 + 128*4

#max8 52*52*128 -> 26*26*128
max4 = max_pool_layer(conv4)

#conv9 26*26*128 -> 26*26*256
w5 = weight_variable([3, 3, 128, 256])
b5 = bias_variable([256])
c5 = conv2(max4, w5) + b5
conv5 = leaky_relu(c5)
n_params = n_params + 3*3*128*256 + 256*4

#max10 26*26*256 -> 13*13*256
max5 = max_pool_layer(conv5)

#conv11 13*13*256 -> 13*13*512
w6 = weight_variable([3, 3, 256, 512])
b6 = bias_variable([512])
c6 = conv2(max5, w6) + b6
conv6 = leaky_relu(c6)
n_params = n_params + 3*3*256*512 + 512*4

#max12 13*13*512 -> 13*13*512
max6 = max_pool_layer(conv6, kernel_size=2, stride=1, padding='SAME')

#conv13 13*13*512 -> 13*13*1024
w7 = weight_variable([3, 3, 512, 1024])
b7 = bias_variable([1024])
c7 = conv2(max6, w7) + b7
conv7 = leaky_relu(c7)
n_params = n_params + 3*3*512*1024 + 1024*4

###############################

#conv14 13*13*1024 -> 13*13*1024
w8 = weight_variable([3, 3, 1024, 1024])
b8 = bias_variable([1024])
c8 = conv2(conv7, w8) + b8
conv8 = leaky_relu(c8)
n_params = n_params + 3*3*1024*1024 + 1024*4

#conv1 13*13*1024 -> 13*13*125
w9 = weight_variable([3, 3, 1024, 125])
b9 = bias_variable([125])
c9 = conv2(conv8, w9) + b9
net = c9
n_params = n_params + 3*3*1024*125 + 125*4

print(n_params)