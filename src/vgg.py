# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io
import pickle
from h5py import File
from src.helpers import pool_layer, conv_layer


_MEAN_PIXEL = tf.constant(np.array([103.939, 116.779, 123.68]).reshape((1, 1, 1, 3)), dtype=tf.float32)


def net(data_path, input_image, reuse=True):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    with File(data_path, 'r') as weights, tf.variable_scope('vgg', reuse=reuse):
        net = {}
        current = input_image
        current = _preprocess(current)
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                block_name = 'block' + name[-3] + '_conv' + name[-1]
                kernels = weights[block_name][block_name + '_W_1:0'][:]
                bias = weights[block_name][block_name + '_b_1:0'][:]
                kernels = tf.get_variable(name + '_W', initializer=tf.constant(kernels, tf.float32))
                bias = tf.get_variable(name + '_b', initializer=tf.constant(bias, tf.float32))
                current = conv_layer(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = pool_layer(current)
            net[name] = current

        assert len(net) == len(layers)
        return net


def _preprocess(image):
    image = image[..., ::-1]
    image -= _MEAN_PIXEL
    return image