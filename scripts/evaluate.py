from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from time import time

import numpy as np
import tensorflow as tf
from scipy.misc import imsave, imread, imresize
import os

from tensorflow.python.client import timeline

from utils import load_image

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def load_graph(file_name):
    with open(file_name, 'rb') as f:
        content = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(content)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def evaluate_graph(graph_file_name, output, x, alpha):
    with load_graph(graph_file_name).as_default() as graph:
        image_buffer_input_x = graph.get_tensor_by_name('input_x:0')
        image_buffer_input_alpha = graph.get_tensor_by_name('input_alpha:0')
        result = graph.get_tensor_by_name('output:0')
    x = imread(x)
    x = imresize(x, (256, 256))
    with tf.device('/cpu'), tf.Session(graph=graph) as sess:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        feed_dict = {
            image_buffer_input_x: [x],
            image_buffer_input_alpha: [alpha]
        }
        tm = []
        start = time()
        r = sess.run(result, feed_dict, options=options, run_metadata=run_metadata)
        tm.append(time() - start)
        print('Inference times:', tm)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(chrome_trace)
    r = np.squeeze(np.clip(r, 0, 255).astype('uint8'))
    imsave(output, r)


def evaluate_graph_magneta(graph_file_name, output, x, alpha):
    with load_graph(graph_file_name).as_default() as graph:
        image_buffer_input_x = graph.get_tensor_by_name('input:0')
        image_buffer_input_style_num = graph.get_tensor_by_name('style_num:0')
        result = graph.get_tensor_by_name('transformer/expand/conv3/conv/Sigmoid:0')
    x = load_image(x, resize=256, random_crop=256).astype('float32')
    style_num = [1] + [0] * 25
    with tf.device('/cpu'), tf.Session(graph=graph) as sess:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        feed_dict = {
            image_buffer_input_x: [x],
            image_buffer_input_style_num: style_num,
        }
        tm = []
        for _ in range(10):
            start = time()
            r = sess.run(result, feed_dict, options=options, run_metadata=run_metadata)
            tm.append(time() - start)
        print('Inference times:', tm)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(chrome_trace)
    r = np.squeeze(np.clip(r, 0, 255).astype('uint8'))
    imsave(output, r)



if __name__ == "__main__":
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument("--graph", type=str, required=True, help='Path to graph')
    train_parser.add_argument("--output", '-o', type=str, required=True, help='Output file')
    train_parser.add_argument("--x", type=str, required=True, help='Path to x')
    train_parser.add_argument("--a", type=float, default=1., help='Alpha')
    train_parser.add_argument("--magneta", action='store_true', default=False)

    args = train_parser.parse_args()
    if args.magneta:
        evaluate_graph_magneta(args.graph, args.output, args.x, args.a)
    else:
        evaluate_graph(args.graph, args.output, args.x.y, args.a)
