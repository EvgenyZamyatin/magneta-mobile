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
from utils import load_image_center_crop

sys.setrecursionlimit(100000)
np.set_printoptions(suppress=True)


def train(args):
    if not os.path.exists(args.output + '/samples'):
        os.makedirs(args.output + '/samples')
    shutil.copy2('./src/model.py', args.output + '/model-save.py')

    with open(args.output + '/args.txt', 'w+') as out:
        out.write(str(args) + os.linesep)
    print('compile')
    image_loader = lambda x: load_image_center_crop(x, 256, True)
    style_img = image_loader(args.style_image)
    model = Model(args.content_weight, args.style_weight, args.tv_weight, args.learning_rate, style_img, args.vgg_path,
                  args.output + '/log')
    if args.weights is not None:
        model.load(args.weights)
    batch_size = args.batch_size
    num_prune_batches = args.prune_steps
    num_batches = args.prune_steps * args.batch_count + args.prune_steps

    with model, \
            BatchGenerator(args.content_images, image_loader, num_batches, batch_size, num_proc=2) as content_bg, \
            BatchGenerator(args.content_images, image_loader, num_prune_batches, args.prune_batch, num_proc=2) as prune_bg:
        print('start train')

        for epoch in range(args.prune_steps):
            model.prune_step(prune_bg.get_batch())

            loss = 0
            cnt = 0
            t = 0
            for _ in tqdm(range(args.batch_count)):
                cnt += 1
                content_batch = content_bg.get_batch()
                alpha = np.random.uniform(0, 1, batch_size)
                t -= time()
                loss = model.train(content_batch, alpha=alpha) + loss
                t += time()

            print('epoch:', epoch)
            print('loss:', loss / cnt)
            print('train time:', t)
            content_batch = content_bg.get_batch()
            samples_1 = [content_batch]
            for alpha in [1.]:
                samples_1.append(model.style(content_batch, alpha=[alpha] * batch_size))
            samples = np.concatenate(samples_1, 1)
            samples = np.concatenate(samples, 1)
            imsave(args.output + '/samples/%04d.png' % epoch, samples)
            model.save(args.output + '/model.pkl')


if __name__ == '__main__':
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument("--prune-steps", type=int, required=True)
    train_parser.add_argument("--prune-batch", type=int, required=True)

    train_parser.add_argument("--content-images", "-ci", type=str, required=True, help='Path to imgs')
    train_parser.add_argument("--style-image", "-si", type=str, required=True, help='Path to img')
    train_parser.add_argument("--vgg-path", "-vgg", type=str, required=True, help='Path to vgg')
    train_parser.add_argument("--content-weight", type=float, default=8)
    train_parser.add_argument("--style-weight", type=float, default=1e-5)
    train_parser.add_argument("--tv-weight", type=float, default=1e-1)
    train_parser.add_argument("--learning-rate", '-lr', type=float, default=1e-4)
    train_parser.add_argument("--batch-size", '-bs', type=int, default=4)
    train_parser.add_argument("--output", "-o", type=str, required=True, help='Path to output')
    train_parser.add_argument("--weights", "-w", type=str, required=True, help='Path to weights')
    train_parser.add_argument("--batch-count", "-bc", type=int, default=1000)

    args = train_parser.parse_args()
    train(args)