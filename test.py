import os

import cv2
import numpy as np
import tensorflow as tf

import tiny_yolo_net as net
import weights_loder

import logging
from logging import info
LOG_FORMAT = "************%(asctime)s - %(levelname)s- %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
#tf.sigmoid()
#tf.nn.softmax()

def iou(boxA, boxB):
    """
    calculate iou value
    input: [x1, y1, x2, y2]
    output : iou value
    """
    #determin the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection
    intersection_area = (xB - xA + 1) * (yB - yA + 1)
    
    # Compute the area of both rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute the IOU = over / union
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou


def preprocessing(input_image_path, net_input_size):
    """
    preprocess input image
    input: input_image_path, net_input_size
    return: preprocessed_image
    """
    input_image = cv2.imread(input_image_path)

    #resize input image to net input size and normalization uint8 -> [0, 1]
    resized_image = cv2.resize(input_image, (net_input_size, net_input_size),  interpolation = cv2.INTER_CUBIC)
    image = np.array([resized_image], dtype='f')
    image /= 255.
    # add the one dimension for batch layer
    image = np.expand_dims(image, 0)
    return image


def postprocessing(predictions, input_image_path, image_size):
    """
    post processing
    input:
        predictions:
        input_image_path: input image path
        image_size: the image size for network input

    """
    input_image = cv2.imread(input_image_path)
    input
    input_iamge = cv2.resize(input_iamge, (image_size, image_size), )

def predict(sess, preprocessed_image):
    """
    predict characteristic using network and output prediction 
    input : sess, preprocessed image
    return : predictions
    """
    predictions = sess.run(net.o9, feed_dict={net.images: preprocessed_image})
    return predictions

    
if __name__ == "__main__":
    input_image_path = 'dog.jpg'
    weights_file = os.path.join('data', 'yolov2-tiny-voc.weights')
    output_image_path = '{}_output.jpg'.format(input_image_path.split('.')[0],input_image_path.split('.')[1])
    ckpt_path = 'ckpt'
    info("input file:{}, output file:{}".format(input_image_path, output_image_path))

    #definiton of the paramters
    net_input_size = 416
    score_thresold = 0.3
    iou_threasold = 0.3

    #definiton sess interactivesession or not session 
    #interactive session can insert some var of options
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #load weights for this sess
    info('loadding ckpt file  or creating ckpt file ')
    saver = tf.train.Saver()
    weights_loder.load(sess, weights_file, ckpt_path, saver)

    #preprocess for input image
    info('prerocessing image')
    preprocessed_image = preprocessing(input_image_path, net_input_size)

    #compute predictions on input image
    info('predicting ...')
    predictions = predict(sess, preprocessed_image)


    


