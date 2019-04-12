import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from model import get_dilated_unet
from keras.utils.np_utils import to_categorical
from model import get_dilated_unet

TEST_SIZE = 1000
WIDTH = 400
HEIGHT = 400

def data_loader():
	# 加载测试数据集，测试数据存放在'../data/test/img'文件夹下
	x_batch = []
	ids_train_batch = np.arange(0, 1000)
	for _id in ids_train_batch:
	    if _id%100 == 0:
	        print('{} imgs done!'.format(_id))
	    img = cv2.imread('../data/test/img/img_{}.jpg'.format(_id))
	    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
	    x_batch.append(img)
	x_batch = np.array(x_batch, np.float32) / 255.

	return x_batch 

def model_loader(model_mode='cascade'):
	weights_path = 'model_weights_{}.hdf5'.format(model_mode)
	model = get_dilated_unet(mode=model_mode)
	model.load_weights(weights_path)
	return model

def main():
	print('data loding ...')
	x_batch = data_loader()

	# cascade mode single model : 0.972
	# parallel mode single model: 0.97
	# ensemble: np.mean(cascade, parallel): 0.9754
	# ensemble with 5 fold result: (提升并不明显，而且5折训练时间太久，实际场景可推广性稍差)
	# 	0.4*single_cascade + 0.3*single_parallel + 0.1*5fold-cascade(select 3 fold)(*3) = 0.976
	print('cascade model loading ...')
	model_cascade = model_loader(model_mode = 'cascade')
	y_batch_cascade = model_cascade.predict(x_batch, batch_size=10)
	print('parallel model loading ...')
	model_parallel = model_loader(model_mode = 'parallel')
	y_batch_parallel = model_parallel.predict(x_batch, batch_size=10)

	# score: 0.9754
	y_test = np.stack((y_batch_cascade, y_batch_parallel), axis=-1)
	y_test = np.mean(y_test, axis=-1)

	print('result saving ...')
	for id in range(TEST_SIZE):
	    res = np.argmax(y_test[id], axis=-1)*60
	    cv2.imwrite('../data/test/label/img_{}.png'.format(id), res)
	print('done!')



if __name__ == '__main__':
	main()