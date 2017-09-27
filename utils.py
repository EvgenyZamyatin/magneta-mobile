import os
import pickle
import re
import time

import numpy as np
from scipy.misc import imread, imresize


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, dirs, files in os.walk(directory) for f in files
            if re.match('([\w]+\.(?:' + ext + '))', f)]


def save(file, obj):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w+b') as out:
        pickle.dump(obj, out)


def load(file):
    with open(file, 'rb') as data:
        return pickle.load(data)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('time: %1.3lfs' % (te - ts))
        return result

    return timed


def load_image(src, resize=512, random_crop=256):
    try:
        img = imread(src)
        if not (len(img.shape) == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))
        if resize is not None:
            h, w = img.shape[:2]
            scale = resize / min(w, h)
            img = imresize(img, (resize, int(np.ceil(w * scale)))) if h < w else imresize(img, (
            int(np.ceil(h * scale)), resize))
        if random_crop is not None:
            h, w = img.shape[:2]
            to_cut_h = h - random_crop
            to_cut_w = w - random_crop
            # print(resize, random_crop, h, w)
            assert to_cut_w >= 0 and to_cut_h >= 0
            x1 = np.random.randint(0, to_cut_h + 1)
            y1 = np.random.randint(0, to_cut_w + 1)
            img = img[x1: x1 + random_crop, y1: y1 + random_crop]
    except Exception as e:
        print(src)
        raise e
    return img


def load_image_center_crop(src, size=None, center_crop=False):
    img = imread(src, mode="RGB")
    if center_crop:
        cur_shape = img.shape[:2]
        shorter_side = min(cur_shape)
        longer_side_xs = max(cur_shape) - shorter_side
        longer_side_start = int(longer_side_xs / 2.)
        longer_side_slice = slice(longer_side_start, longer_side_start + shorter_side)
        if shorter_side == cur_shape[0]:
            img = img[:, longer_side_slice, :]
        else:
            img = img[longer_side_slice, :, :]

    if size is not None:
        cur_shape = img.shape[:2]
        shorter_side = min(cur_shape)
        aspect = max(cur_shape) / float(shorter_side)
        new_shorter_side = int(size / aspect)
        if shorter_side == cur_shape[0]:
            new_shape = (new_shorter_side, size)
        else:
            new_shape = (size, new_shorter_side)
        img = imresize(img, new_shape)
    return img