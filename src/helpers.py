import tensorflow as tf
from tensorflow.contrib import slim as slim


def pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME')


def conv_layer(input, weights, bias, pad='SAME'):
    conv = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1),
                        padding=pad)
    return tf.nn.bias_add(conv, bias)


def conv_block(input, name, features, filter_size, stride, relu=True, norm=True, **kwargs):
    with tf.variable_scope(name):
        _, rows, cols, in_channels = [i.value for i in input.get_shape()]
        weights = tf.get_variable('W',
                                  shape=[filter_size, filter_size, in_channels, features],
                                  initializer=tf.contrib.layers.xavier_initializer())
        #pad = filter_size // 2
        #input = tf.pad(input, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(input, weights, (1, stride, stride, 1), 'SAME')

        if norm:
            conv = _instance_norm(conv, name+'/norm')
        else:
            bias = tf.get_variable('b', initializer=tf.zeros(features, tf.float32))
            conv = tf.nn.bias_add(conv, bias)

        if relu:
            conv = tf.nn.relu(conv)
    return conv


def conv_block_mob(input, name, features, filter_size, stride, relu=True, norm=True, sep_conv=True, low_rank=False, **kwargs):
    with tf.variable_scope(name):
        _, rows, cols, in_channels = [i.value for i in input.get_shape()]

        if low_rank:
            depthwise_weights_w = tf.get_variable('W_depth_w',
                                                shape=[1, filter_size, in_channels, 1],
                                                initializer=tf.contrib.layers.xavier_initializer())
            depthwise_weights_h = tf.get_variable('W_depth_h',
                                                shape=[filter_size, 1, in_channels, 1],
                                                initializer=tf.contrib.layers.xavier_initializer())
        else:
            depthwise_weights = tf.get_variable('W_depth',
                                                  shape=[filter_size, filter_size, in_channels, 1],
                                                  initializer=tf.contrib.layers.xavier_initializer())

        pointwise_weights = tf.get_variable('W_point',
                                            shape=[1, 1, in_channels, features],
                                            initializer=tf.contrib.layers.xavier_initializer())

        if not low_rank:
            weights = tf.einsum('abcd,efcg->abcg', depthwise_weights, pointwise_weights)

            conv = input
            if sep_conv:
                conv = tf.nn.depthwise_conv2d(conv, depthwise_weights, (1, stride, stride, 1), 'SAME')
                conv = tf.nn.conv2d(conv, pointwise_weights, (1, 1, 1, 1), 'VALID')
            else:
                conv = tf.nn.conv2d(conv, weights, (1, stride, stride, 1), 'SAME')
        else:
            conv = input
            conv = tf.nn.depthwise_conv2d(conv, depthwise_weights_w, (1, 1, stride, 1), 'SAME')
            conv = tf.nn.depthwise_conv2d(conv, depthwise_weights_h, (1, stride, 1, 1), 'SAME')
            conv = tf.nn.conv2d(conv, pointwise_weights, (1, 1, 1, 1), 'VALID')

        if norm:
            conv = _instance_norm(conv, name+'/norm')
        else:
            bias = tf.get_variable('b', initializer=tf.zeros(features))
            conv = tf.nn.bias_add(conv, bias)
        if relu:
            conv = tf.nn.relu(conv)
    return conv


def upsampling(input_, name, features, filter_size, stride, relu=True, norm=True, **kwargs):
    with tf.variable_scope(name):
        _, height, width, _ = [s.value for s in input_.get_shape()]
        upsampled_input = tf.image.resize_nearest_neighbor(input_, [stride * height, stride * width])
        return conv_block(upsampled_input, name + '/conv', features, filter_size, 1, relu, norm)


def upsampling_mob(input_, name, features, filter_size, stride, relu=True, norm=True, sep_conv=True, low_rank=False, **kwargs):
    with tf.variable_scope(name):
        _, height, width, _ = [s.value for s in input_.get_shape()]
        upsampled_input = tf.image.resize_nearest_neighbor(input_, [stride * height, stride * width])
        return conv_block_mob(upsampled_input, name + '/conv', features, filter_size, 1, relu, norm, sep_conv, low_rank)


def residual_block(input_, name, kernel_size, **kwargs):
    with tf.variable_scope(name):
        num_outputs = input_.get_shape()[-1].value
        h_1 = conv_block(input_, name + '/conv1', num_outputs, kernel_size, 1)
        h_2 = conv_block(h_1, name + '/conv2', num_outputs, kernel_size, 1, False, False)
        return input_ + h_2


def residual_block_mob(input_, name, kernel_size, sep_conv=True, **kwargs):
    with tf.variable_scope(name):
        num_outputs = input_.get_shape()[-1].value
        h_1 = conv_block_mob(input_, name + '/conv1', num_outputs, kernel_size, 1, True, True, sep_conv)
        h_2 = conv_block_mob(h_1, name + '/conv2', num_outputs, kernel_size, 1, False, False, sep_conv)
        return input_ + h_2


def moments(x, axis):
    mean = tf.reduce_mean(x, axis, keep_dims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis, keep_dims=True)
    return mean, var


def _instance_norm(x, name):
    with tf.variable_scope(name):
        _, rows, cols, in_channels = [i.value for i in x.get_shape()]
        mu = tf.get_variable('mu',
                             initializer=tf.zeros(in_channels, tf.float32))
        sigma = tf.get_variable('sigma',
                                initializer=tf.ones(in_channels, tf.float32))
        x_mu, x_sigma_sq = moments(x, (1, 2))
        epsilon = 1e-5
        y = tf.nn.batch_normalization(
            x, x_mu, x_sigma_sq, mu, sigma, epsilon)
        #x_normalized = (x - x_mu) / tf.sqrt(x_sigma_sq + epsilon)
        #y = x_normalized * sigma + mu
        return y


"""
def nearest_up_sampling(net, scale=2):
    net = tf.image.resize_nearest_neighbor(net, (net.shape[1].value * scale, net.shape[2].value * scale))
    return net

"""
