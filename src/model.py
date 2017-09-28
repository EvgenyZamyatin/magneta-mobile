from collections import OrderedDict

import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

from src.vgg import net as vgg_net
from src.helpers import *
import os

STYLE_LAYERS_VGG = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER_VGG = 'relu4_2'


def _preprocess(x):
    x = np.array(x)
    return x.astype('float32')


def _unprocess(x):
    return np.clip(x, 0, 255).astype('uint8')


def _gram_matrix(feature_maps):
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.to_float(height * width)
    feature_maps = tf.reshape(
        feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator


class Model:
    def __init__(self, content_weight, style_weight, tv_weight, learning_rate, style_image, vgg_path=None,
                 log_path=None, **kwargs):
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.content_weight = content_weight
        self.learning_rate = learning_rate
        self.log_path = log_path
        self.style_image = _preprocess(style_image)
        self.sess = None
        self.weights = None
        self.writer = None
        self.seen_batch_count = 0
        self.vgg_path = vgg_path
        self._build_encoder_vgg(vgg_path)
        self._build_transformer()
        self._build_fn()

    def __enter__(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.weights:
            self.saver.restore(self.sess, self.weights)
        if self.log_path:
            self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
        if self.writer:
            self.writer.close()
        self.sess = None
        self.writer = None

    def style(self, x, **kwargs):
        x = _preprocess(x)
        args = {
            self.args['x']: x,
        }
        res = self.sess.run(self.test, feed_dict=args)
        return _unprocess(res)

    def train(self, x, **kwargs):
        x = _preprocess(x)
        args = {
            self.args['x']: x,
        }
        _, summary, cl, sl, tl = self.sess.run([self.train_step, self.summary_op] + self.losses, feed_dict=args)
        res = np.array([cl, sl, tl])
        self.seen_batch_count += 1
        self.writer.add_summary(summary, self.seen_batch_count)
        assert not np.isnan(res.sum())
        return res

    def _build_encoder_vgg(self, vgg_path):
        self._encode_vgg = lambda x, reuse: vgg_net(vgg_path, x, reuse)

    def _build_transformer(self):
        def transform(input_, reuse, sep_conv, only_out=True):
            with tf.variable_scope('transformer', reuse=reuse):
                net = {}
                input_ /= 255
                net['input'] = input_
                h = conv_block_mob(input_, 'conv1', 16, 3, 1, sep_conv=sep_conv)
                net['conv1'] = h
                #h *= tf.get_variable('conv1_prune', initializer=tf.ones(16), trainable=False)
                net['conv1_pruned'] = h
                h = conv_block_mob(h, 'conv2', 32, 3, 2, sep_conv=sep_conv)
                net['conv2'] = h
                #h *= tf.get_variable('conv2_prune', initializer=tf.ones(32), trainable=False)
                net['conv2_pruned'] = h
                h = conv_block_mob(h, 'conv3', 64, 3, 2, sep_conv=sep_conv)
                net['conv3'] = h
                #h *= tf.get_variable('conv3_prune', initializer=tf.ones(64), trainable=False)
                net['conv3_pruned'] = h

                h = residual_block_mob(h, 'residual1', 3, sep_conv=sep_conv)
                net['residual1'] = h
                #h *= tf.get_variable('residual1_prune', initializer=tf.ones(64), trainable=False)
                net['residual1_pruned'] = h
                h = residual_block_mob(h, 'residual2', 3, sep_conv=sep_conv)
                net['residual2'] = h
                #h *= tf.get_variable('residual2_prune', initializer=tf.ones(64), trainable=False)
                net['residual2_pruned'] = h
                h = residual_block_mob(h, 'residual3', 3, sep_conv=sep_conv)
                net['residual3'] = h
                #h *= tf.get_variable('residual3_prune', initializer=tf.ones(64), trainable=False)
                net['residual3_pruned'] = h

                h = upsampling_mob(h, 'uconv2', 32, 3, 2, sep_conv=sep_conv)
                net['uconv2'] = h
                #h *= tf.get_variable('uconv2_prune', initializer=tf.ones(32), trainable=False)
                net['uconv2_pruned'] = h
                h = upsampling_mob(h, 'uconv3', 16, 3, 2, sep_conv=sep_conv)
                net['uconv3'] = h
                #h *= tf.get_variable('uconv3_prune', initializer=tf.ones(16), trainable=False)
                net['uconv3_pruned'] = h

                h = upsampling_mob(h, 'uconv4', 3, 3, 1, False, False, sep_conv=sep_conv)
                net['uconv4'] = h
                h = tf.nn.tanh(h) * 150 + 255 / 2
            if only_out:
                return h
            return h, net
        self._transform = transform

    def _build_fn(self):
        x_var = tf.placeholder(tf.float32, (None, 256, 256, 3), 'input_x')
        r, transform_net = self._transform(x_var, False, False, False)
        x_ftr_vgg = self._encode_vgg(x_var, False)
        r_ftr_vgg = self._encode_vgg(r, True)

        fr_vgg = r_ftr_vgg[CONTENT_LAYER_VGG]
        fx_vgg = x_ftr_vgg[CONTENT_LAYER_VGG]

        r_style_f = [r_ftr_vgg[i] for i in STYLE_LAYERS_VGG]
        x_style_f = [x_ftr_vgg[i] for i in STYLE_LAYERS_VGG]
        with self:
            y_style_f = self.sess.run([x_ftr_vgg[i] for i in STYLE_LAYERS_VGG],
                                      {x_var: self.style_image[np.newaxis]})

        content_loss = tf.nn.l2_loss(fr_vgg - fx_vgg) / tf.cast(tf.size(fr_vgg), tf.float32)

        style_loss = 0
        for _, yf, rf in zip(x_style_f, y_style_f, r_style_f):
            y_gram = _gram_matrix(yf)
            r_gram = _gram_matrix(rf)
            style_loss += tf.nn.l2_loss(y_gram - r_gram) / tf.cast(tf.size(y_gram), tf.float32)
        y_tv = tf.nn.l2_loss(r[:, 1:, :, :] - r[:, :-1, :, :]) / tf.cast(tf.size(r[:, 1:, :, :]), tf.float32)
        x_tv = tf.nn.l2_loss(r[:, :, 1:, :] - r[:, :, :-1, :]) / tf.cast(tf.size(r[:, :, 1:, :]), tf.float32)
        tv_loss = (x_tv + y_tv)
        losses = [self.content_weight * content_loss, self.style_weight * style_loss, self.tv_weight * tv_loss]
        loss = sum(losses)

        tf.summary.scalar("content_loss", losses[0])
        tf.summary.scalar("style_loss", losses[1])
        tf.summary.scalar("tv_loss", losses[2])
        self.summary_op = tf.summary.merge_all()
        self.losses = losses

        decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'transformer')
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=decoder_params)
        self.test = tf.clip_by_value(self._transform(x_var, True, True), 0, 255, 'output')
        self.args = {
            'x': x_var,
        }
        self.saver = tf.train.Saver()

        rank = OrderedDict()
        for layer in ['conv1_pruned', 'conv2_pruned', 'conv3_pruned',
                      'residual1_pruned', 'residual2_pruned', 'residual3_pruned',
                      'uconv2_pruned', 'uconv3_pruned']:
            h = transform_net[layer]
            cur_rank = tf.reduce_mean(h * tf.gradients(content_loss, h), axis=(0, 1, 2))
            print(cur_rank.shape)
            cur_rank = tf.nn.l2_normalize(cur_rank, 0)
            rank[layer] = cur_rank
        self.rank = rank

    def prune_step(self, batch):
        batch = _preprocess(batch)
        for layer, f in self.rank.items():
            res = self.sess.run(f, feed_dict={self.args['x']: batch})
            print(layer)
            print(res)
            print((res**2).sum())
            print(res.mean())
            print(res.shape)

    def save(self, file):
        assert self.sess
        self.saver.save(self.sess, file)
        self.save_graph_to_file(os.path.dirname(file) + '/graph.pb')

    def load(self, file):
        self.weights = file

    def save_graph_to_file(self, graph_file_name):
        output_graph_def = graph_util.convert_variables_to_constants(
            self.sess, self.sess.graph.as_graph_def(), ['output'])
        with gfile.FastGFile(graph_file_name, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
