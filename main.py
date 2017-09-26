import argparse
import os
import shutil
import sys
from time import time

import numpy as np
from scipy.misc import imsave, imread, imresize
from tqdm import tqdm

from batch_generator import BatchGenerator
from src.model import Model

sys.setrecursionlimit(100000)
np.set_printoptions(suppress=True)


def train(args):
    if not os.path.exists(args.output + '/samples'):
        os.makedirs(args.output + '/samples')
    shutil.copy2('./src/model.py', args.output + '/model-save.py')
    with open(args.output + '/args.txt', 'w+') as out:
        out.write(str(args) + os.linesep)
    print('compile')

    style_img = imread(args.style_image)
    style_img = imresize(style_img, (256, 256))

    model = Model(args.content_weight, args.style_weight, args.tv_weight, args.learning_rate, style_img, args.vgg_path,
                  args.output + '/log')
    if args.weights is not None:
        model.load(args.weights)
    batch_size = args.batch_size
    num_batches = args.epoch_count * args.batch_count + args.epoch_count
    with model, \
            BatchGenerator(args.content_images, num_batches, batch_size, 512, 256, num_proc=4) as content_bg:
        print('start train')
        for epoch in range(args.start_epoch, args.epoch_count):
            loss = 0
            cnt = 0
            t = 0
            for _ in tqdm(range(args.batch_count)):
                cnt += 1
                content_batch = content_bg.get_batch()
                alpha = np.random.uniform(0, 1, batch_size)
                t -= time()
                loss = model.train(content_batch, alpha) + loss
                t += time()

            print('epoch:', epoch)
            print('loss:', loss / cnt)
            print('train time:', t)
            alpha = np.random.uniform(0, 1, batch_size)
            content_batch = content_bg.get_batch()
            samples_0 = content_batch
            samples_1 = model.style(content_batch, alpha)
            samples = np.concatenate([samples_0, samples_1], 2)
            samples = np.concatenate(samples, axis=0)
            imsave(args.output + '/samples/%04d.png' % epoch, samples)
            model.save(args.output + '/model.pkl')


if __name__ == '__main__':
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument("--content-images", "-ci", type=str, required=True, help='Path to imgs')
    train_parser.add_argument("--style-image", "-si", type=str, required=True, help='Path to img')
    train_parser.add_argument("--vgg-path", "-vgg", type=str, required=True, help='Path to vgg')
    train_parser.add_argument("--content-weight", type=float, default=1)
    train_parser.add_argument("--style-weight", type=float, default=1)
    train_parser.add_argument("--tv-weight", type=float, default=1)
    train_parser.add_argument("--learning-rate", '-lr', type=float, default=1e-4)
    train_parser.add_argument("--batch-size", '-bs', type=int, default=4)
    train_parser.add_argument("--output", "-o", type=str, required=True, help='Path to output')
    train_parser.add_argument("--weights", "-w", type=str, default=None, help='Path to weights')
    train_parser.add_argument("--epoch-count", "-ec", type=int, default=40)
    train_parser.add_argument("--start-epoch", "-se", type=int, default=0)
    train_parser.add_argument("--batch-count", "-bc", type=int, default=1000)

    args = train_parser.parse_args()
    train(args)