import threading
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from model import get_dilated_unet
from keras.utils.np_utils import to_categorical
from model import get_dilated_unet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_SIZE = 50000
BATCH_SIZE = 5 
WIDTH = 400
HEIGHT = 400
log_path = 'log.csv'
weights_path = 'model_weights_cascade.hdf5' 
# weights_path = 'model_weights_parallel.hdf5'


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
def train_generator(IMG_SIZE):
	while True:
		shuffle_indices = np.arange(IMG_SIZE)
		shuffle_indices = np.random.permutation(shuffle_indices)
		
		for start in range(0, IMG_SIZE, BATCH_SIZE):
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
def valid_generator(IMG_SIZE):
	while True:
		for start in range(0, IMG_SIZE, BATCH_SIZE):
			x_batch = []
			y_batch = []

			end = min(start + BATCH_SIZE, IMG_SIZE)
			ids_train_batch = np.arange(start, end)

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
	model = get_dilated_unet(mode='cascade') # 线上0.972 （Ps：未完全迭代情况下）
	# model = get_dilated_unet(mode='parallel') # 线上0.97 （Ps：未完全迭代情况下）
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

	model.fit_generator(generator=train_generator(IMG_SIZE),
                    steps_per_epoch=np.ceil(float(IMG_SIZE) / float(BATCH_SIZE)),
                    epochs=50,
                    callbacks=callbacks,
                    validation_data=valid_generator(IMG_SIZE),
                    validation_steps=np.ceil(float(IMG_SIZE)) / float(BATCH_SIZE))


if __name__ == '__main__':
	main()