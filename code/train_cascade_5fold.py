#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm
import seaborn as sns
import pickle
import glob
from keras.preprocessing.image import img_to_array
from keras import layers
from keras import models
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras import backend as keras
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.utils.np_utils import to_categorical
import threading
from sklearn.model_selection import train_test_split, KFold
from model import get_dilated_unet

# 指定GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



IMG_SIZE = 50000
BATCH_SIZE = 5 
WIDTH = 400
HEIGHT = 400


class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(shuffle_indices):
    while True:
        for start in range(0, len(shuffle_indices), BATCH_SIZE):
            x_batch = []
            y_batch = []
            
            end = min(start + BATCH_SIZE, IMG_SIZE)
            ids_train_batch = shuffle_indices[start:end]
            
            for _id in ids_train_batch:
                img = cv2.imread('../data/train/img/img_{}.jpg'.format(_id))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = cv2.imread('../data/train/label/label_{}.png'.format(_id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                x_batch.append(img)
                y_batch.append(mask)
            
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.int) // 60
            y_batch = to_categorical(y_batch, 4)
            
            yield x_batch, y_batch

@threadsafe_generator
def valid_generator(shuffle_indices):
    while True:
        for start in range(0, len(shuffle_indices), BATCH_SIZE):
            x_batch = []
            y_batch = []

            end = min(start + BATCH_SIZE, IMG_SIZE)
            ids_train_batch = shuffle_indices[start:end]

            for _id in ids_train_batch:
                img = cv2.imread('../data/train/img/img_{}.jpg'.format(_id))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

                mask = cv2.imread('../data/train/label/label_{}.png'.format(_id),
                                  cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.int) // 60
            y_batch = to_categorical(y_batch, 4)

            yield x_batch, y_batch


def main():
	folds = list(KFold(n_splits=5, shuffle=True, random_state=1).split(range(50000)))

	for j, (train_idx, val_idx) in enumerate(folds):
		print('\nFold ' + str(j))
		weights_path = 'cascade_fold' + str(j) + '_weights.hdf5'
		log_path = 'log_' + str(j) + '.csv'
		train_idx = list(train_idx)
		val_idx = list(val_idx)
		model = get_dilated_unet()
		callbacks = [EarlyStopping(monitor='val_loss',
								   patience=10,
								   verbose=1,
								   min_delta=1e-4,
								   mode='min'),
					 ReduceLROnPlateau(monitor='val_loss',
									   factor=0.2,
									   patience=5,
									   verbose=1,
									   epsilon=1e-4,
									   mode='min'),
					 ModelCheckpoint(monitor='val_loss',
									 filepath=weights_path,
									 save_best_only=True,
									 mode='min'),
					 CSVLogger(log_path, append=True, separator=';')]


		model.fit_generator(generator=train_generator(train_idx),
						steps_per_epoch=np.ceil(float(len(train_idx)) / float(BATCH_SIZE)),
						epochs=50,
						callbacks=callbacks,
						validation_data=valid_generator(val_idx),
						validation_steps=np.ceil(float(len(val_idx))) / float(BATCH_SIZE))


if __name__ == '__main__':
    main()