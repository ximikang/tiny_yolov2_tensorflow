import os
import os.path
import numpy as np
import tensorflow as tf
import tiny_yolo_net

import logging
from logging import info
LOG_FORMAT = "************%(asctime)s - %(levelname)s- %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def load_conv_layer_bn(name, loaded_weights, shape, offset):
    """
    load parameters for bn layer
    return biases, kernel_wights, offset
    """
    n_kernel_weights = shape[0]*shape[1]*shape[2]*shape[3]
    n_output_channels = shape[-1]

    n_bn_mean = n_output_channels
    n_bn_var = n_output_channels
    n_bn_biases = n_output_channels
    n_bn_gamma = n_output_channels

    n_weights_conb_bn = (n_kernel_weights + n_output_channels * 4)

    """
    n_kernel_wights + n_output_channels*4

    kernel_shape + n_biases + n_bn_means + n_bn_var + n_bn_gammas

    """

    biases = loaded_weights[offset:offset + n_bn_biases]
    offset = offset + n_bn_biases
    gammas = loaded_weights[offset:offset + n_bn_gamma]
    offset = offset + n_bn_gamma
    means = loaded_weights[offset:offset + n_bn_mean]
    offset = offset + n_bn_mean
    var = loaded_weights[offset:offset + n_bn_var]
    offset = offset + n_bn_var
    kernel_weights = loaded_weights[offset:offset + n_kernel_weights]
    offset = offset + n_kernel_weights

    # darktiny_yolo_net conv_weights (out_dim, in_dim, heights, width)
    kernel_weights = np.reshape(kernel_weights, (shape[3],shape[2],shape[0],shape[1]), order='C')

    # solve batch normalizaiton parameters
    for i in range(n_output_channels):

        scale = gammas[i] / np.sqrt(var[i] + tiny_yolo_net.bn_epsilon)
        kernel_weights[i,:,:,:] = kernel_weights[i,:,:,:] * scale
        biases[i] = biases[i] - means[i]*scale

    kernel_weights = np.transpose(kernel_weights, [2,3,1,0])
    return biases, kernel_weights, offset


def load_conv_layer(name, loaded_weights, shape, offset):
    n_kernel_weights = shape[0]*shape[1]*shape[2]*shape[3]
    n_output_channels = shape[-1]
    n_biases = n_output_channels

    n_weight_conv = (n_kernel_weights, n_output_channels)
    
    biases = loaded_weights[offset:offset+n_biases]
    offset = offset + n_biases
    kernel_weights  = loaded_weights[offset:offset + n_kernel_weights]
    offset = offset +n_kernel_weights

    kernel_weights = np.reshape(kernel_weights,(shape[3],shape[2],shape[0],shape[1]),order='C')
    kernel_weights = np.transpose(kernel_weights,[2,3,1,0])
    
    return biases, kernel_weights , offset


def load(sess, weights_file, ckpt_path, saver):

    if(os.path.exists(ckpt_path)):
        info("found ckpt file")
        checkpoint_files_path = os.path.join(ckpt_path,"model.ckpt")
        saver.restore(sess, checkpoint_files_path)
        info("loaded weights from ckpt")

    loaded_weights = []
    loaded_weights = np.fromfile(weights_file, dtype='f')
    # delete the first four beacuse there are not real params
    loaded_weights = loaded_weights[4:]

    
    info('total number of params = {}'.format(len(loaded_weights)))
    offset = 0


    # conv1 3*3*3*16
    biases, kernel_weights, offset = load_conv_layer_bn('conv1', loaded_weights, [3,3,3,16], offset)
    sess.run(tf.assign(tiny_yolo_net.b1, biases))
    sess.run(tf.assign(tiny_yolo_net.w1, kernel_weights))

    # conv2 3*3*16*32
    biases, kernel_weights, offset = load_conv_layer_bn('conv2', loaded_weights, [3,3,16,32], offset)
    sess.run(tf.assign(tiny_yolo_net.b2, biases))
    sess.run(tf.assign(tiny_yolo_net.w2, kernel_weights))

    # conv3 3*3*32*64
    biases, kernel_weights, offset = load_conv_layer_bn('conv3', loaded_weights, [3,3,32,64], offset)
    sess.run(tf.assign(tiny_yolo_net.b3, biases))
    sess.run(tf.assign(tiny_yolo_net.w3, kernel_weights))

    # conv4 3*3*64*128
    biases, kernel_weights, offset = load_conv_layer_bn('conv4', loaded_weights, [3,3,64,128], offset)
    sess.run(tf.assign(tiny_yolo_net.b4, biases))
    sess.run(tf.assign(tiny_yolo_net.w4, kernel_weights))

    # conv5 3*3*128*256
    biases, kernel_weights, offset = load_conv_layer_bn('conv5', loaded_weights, [3,3,128,256], offset)
    sess.run(tf.assign(tiny_yolo_net.b5, biases))
    sess.run(tf.assign(tiny_yolo_net.w5, kernel_weights))

    # conv6 3*3*256*512
    biases, kernel_weights, offset = load_conv_layer_bn('conv6', loaded_weights, [3,3,256,512], offset)
    sess.run(tf.assign(tiny_yolo_net.b6, biases))
    sess.run(tf.assign(tiny_yolo_net.w6, kernel_weights))

    # conv7 3*3*512*1024
    biases, kernel_weights, offset = load_conv_layer_bn('conv7', loaded_weights, [3,3,512,1024], offset)
    sess.run(tf.assign(tiny_yolo_net.b7, biases))
    sess.run(tf.assign(tiny_yolo_net.w7, kernel_weights))

    # conv8 3*3*1024*1024
    biases, kernel_weights, offset = load_conv_layer_bn('conv8', loaded_weights, [3,3,1024,1024], offset)
    sess.run(tf.assign(tiny_yolo_net.b8, biases))
    sess.run(tf.assign(tiny_yolo_net.w8, kernel_weights))

    # conv9 1*1*1024*125 no bn layer
    biases, kernel_weights, offset = load_conv_layer('conv9', loaded_weights, [1,1,1024,125], offset)
    sess.run(tf.assign(tiny_yolo_net.b9, biases))
    sess.run(tf.assign(tiny_yolo_net.w9, kernel_weights))


    info("final offset:{}".format(offset))
    info("total number of params in weights file = {}".format(len(loaded_weights)))

    if not os.path.exists(ckpt_path):
        info("not ckpt file, then will create a new .ckpt file:{}".format(ckpt_path))
        os.makedirs(ckpt_path)
        checkpoint_files_path = os.path.join(ckpt_path, "model.ckpt")
        saver.save(sess,checkpoint_files_path)

if __name__ == "__main__":
    sess = tf.Session()
    saver = tf.train.Saver()
    weights_file = os.path.join('data', 'yolov2-tiny-voc.weights')
    ckpt_path = os.path.join('ckpt')
    load(sess, weights_file, ckpt_path, saver)