# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import pickle
import numpy as np
import random
import os
from util.uitls import pad_seq, bytes_to_file, read_split_image, shift_and_resize_image, normalize_image


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


def get_batch_iter(examples, batch_size, augment):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def batch_iter():
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            processed = [process(e, augment) for e in batch]
            # stack into tensor
            yield np.array(processed).astype(np.float32)

    return batch_iter()


def process(img, augment):
    img = bytes_to_file(img)
    try:
        img_A, img_B = read_split_image(img)
        if augment:
            # augment the image by:
            # 1) enlarge the image
            # 2) random crop the image back to its original size
            # NOTE: image A and B needs to be in sync as how much
            # to be shifted
            w, h = img_A.shape
            multiplier = random.uniform(1.00, 1.20)
            # add an eps to prevent cropping issue
            nw = int(multiplier * w) + 1
            nh = int(multiplier * h) + 1
            shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
            shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
            img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
            img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)

        img_A = normalize_image(img_A)
        img_B = normalize_image(img_B)

        # 2D to 3D matrix
        img_A = np.reshape(img_A, [img_A.shape[0], img_A.shape[1], 1])
        img_B = np.reshape(img_B, [img_B.shape[0], img_B.shape[1], 1])

        return np.concatenate([img_A, img_B], axis=2)
    finally:
        img.close()


class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj"):
        self.data_dir = data_dir

        self.train = PickledImageProvider(os.path.join(self.data_dir, train_name))
        self.val = PickledImageProvider(os.path.join(self.data_dir, val_name))
        print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, augment=True)

    def get_val_iter(self, batch_size, shuffle=True):
        val_examples = self.val.examples[:]

        if shuffle:
            np.random.shuffle(val_examples)
        return get_batch_iter(val_examples, batch_size, augment=False)

    def get_train_sample(self, size, shuffle=False):
        train_samples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(train_samples)

        train_samples = train_samples[:size]
        processed = [process(e, augment=False) for e in train_samples]

        return np.array(processed).astype(np.float32)

    def get_val(self, size, shuffle=False):
        val_examples = self.val.examples[:]

        if shuffle:
            np.random.shuffle(val_examples)

        val_examples = val_examples[0: size]

        processed = [process(e, augment=False) for e in val_examples]

        return np.array(processed).astype(np.float32)

    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))


class InjectDataProvider(object):
    def __init__(self, obj_path):
        self.data = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.data.examples))

    def get_iter(self, batch_size):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for images in batch_iter:
            yield images

