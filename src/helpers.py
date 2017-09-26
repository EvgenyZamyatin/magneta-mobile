import tensorflow as tf
from tensorflow.contrib import slim as slim


def conv_layer(input, weights, bias, pad='SAME'):
    conv = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1),
                        padding=pad)
    return tf.nn.bias_add(conv, bias)


def conv_block(input, alpha, name, features, filter_size, stride, relu=True, norm=True, **kwargs):
    with tf.variable_scope(name):
        _, rows, cols, in_channels = [i.value for i in input.get_shape()]
        weights = tf.get_variable('W',
                                  shape=[filter_size, filter_size, in_channels, features],
                                  initializer=tf.contrib.layers.xavier_initializer())

        conv = tf.nn.conv2d(input, weights, (1, stride, stride, 1), 'SAME')
        if norm:
            conv = _cond_instance_norm(conv, alpha, name+'/norm')
        else:
            bias = tf.get_variable('b',
                                    initializer=tf.zeros(features, tf.float32))
            conv = tf.nn.bias_add(conv, bias)
        if relu:
            conv = tf.nn.relu(conv)
    return conv


def upsampling(input_, alpha, name, features, filter_size, stride, relu=True, norm=True):
    with tf.variable_scope(name):
        _, height, width, _ = [s.value for s in input_.get_shape()]
        upsampled_input = tf.image.resize_nearest_neighbor(
        input_, [stride * height, stride * width])
        return conv_block(upsampled_input, alpha, name + '/conv', features, filter_size, 1, relu, norm)


def residual_block(input_, alpha, name, kernel_size):
    with tf.variable_scope(name):
        num_outputs = input_.get_shape()[-1].value
        h_1 = conv_block(input_, alpha, name + '/conv1', num_outputs, kernel_size, 1)
        h_2 = conv_block(h_1, alpha, name + '/conv2', num_outputs, kernel_size, 1)
        return input_ + h_2


def moments(x, axis):
    mean = tf.reduce_mean(x, axis, keep_dims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis, keep_dims=True)
    return mean, var


def _cond_instance_norm(x, alpha, name):
    with tf.variable_scope(name):
        _, rows, cols, in_channels = [i.value for i in x.get_shape()]
        mu = tf.get_variable('mu',
                             initializer=tf.zeros(in_channels, tf.float32))
        sigma = tf.get_variable('sigma',
                                initializer=tf.ones(in_channels, tf.float32))

        x_mu, x_sigma_sq = moments(x, (1, 2))
        epsilon = 1e-5
        x_normalized = (x - x_mu) / tf.sqrt(x_sigma_sq + epsilon)
        y = x_normalized * sigma + mu
        result = x * (1 - alpha) + y * alpha
        return result


"""
def conv_wn_block_mob(input, name, features, filter_size, relu=True, sep_conv=True):
    with tf.variable_scope(name):
        _, rows, cols, in_channels = [i.value for i in input.get_shape()]

        depthwise_weights = tf.get_variable('W_depth',
                                            shape=[filter_size, filter_size, in_channels, 1],
                                            initializer=tf.contrib.layers.xavier_initializer())

        pointwise_weights = tf.get_variable('W_point',
                                            shape=[1, 1, in_channels, features],
                                            initializer=tf.contrib.layers.xavier_initializer())

        weights = tf.einsum('abcd,efcg->abcg', depthwise_weights, pointwise_weights)
        normalization = tf.sqrt(tf.reduce_sum(tf.square(weights), (0, 1, 2), True)) + 1e-5
        bias = tf.get_variable('b', initializer=tf.zeros(features))

        conv = input
        if sep_conv:
            conv = tf.nn.depthwise_conv2d(conv, depthwise_weights, (1, 1, 1, 1), 'SAME')
            conv = tf.nn.conv2d(conv, pointwise_weights, (1, 1, 1, 1), 'SAME')
        else:
            conv = tf.nn.conv2d(conv, weights, (1, 1, 1, 1), 'SAME')

        conv /= normalization
        conv = tf.nn.bias_add(conv, bias)

        if relu:
            conv = tf.nn.relu(conv)

    return conv
"""


def nearest_up_sampling(net, scale=2):
    net = tf.image.resize_nearest_neighbor(net, (net.shape[1].value * scale, net.shape[2].value * scale))
    return net


def pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME')
