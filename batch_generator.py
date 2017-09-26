import sys
from multiprocessing import Process, Queue
from numpy.random import choice
import numpy as np

from utils import load_image, list_pictures

DEFAULT_MAX_QSIZE = 1000


class BatchGenerator:

    def __init__(self, imdir, num_batches, batch_size, resize, crop, max_qsize=None, num_proc=1):
        max_qsize = max_qsize if max_qsize is not None else DEFAULT_MAX_QSIZE
        self.batchq = Queue(max_qsize)
        self.generator_processes = [Process(target=BatchGenerator.generate_batches, args=(self.batchq, imdir, num_batches, batch_size, resize, crop)) for _ in range(num_proc)]
        self.consumed_batches = 0
        self.num_batches = num_batches

    def get_batch(self):
        if self.consumed_batches == self.num_batches:
            raise StopIteration
        else:
            self.consumed_batches += 1
            return self.batchq.get()

    def __enter__(self):
        for generator_process in self.generator_processes:
            generator_process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for generator_process in self.generator_processes:
            generator_process.terminate()
            generator_process.join()

    @staticmethod
    def generate_batches(batchq, imdir, num_batches, batch_size, resize, crop):
        image_paths = list_pictures(imdir)
        if not image_paths:
            print("Error: no images found in {}".format(imdir))
            sys.exit(1)
        for _  in range(num_batches):
            batch_image_paths = choice(image_paths, batch_size)
            batch = np.vstack([load_image(image_path, resize, crop)[np.newaxis] for image_path in batch_image_paths])
            batchq.put(batch)