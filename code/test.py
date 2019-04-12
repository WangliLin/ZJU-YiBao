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
	# 加载测试数据集，测试数据存放在'../data/test/img/'文件夹下
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

def model_loader(model_mode):
	weights_path = 'model_weights_{}.hdf5'.format(model_mode)
	model = get_dilated_unet(mode=model_mode)
	model.load_weights(weights_path)
	return model

def main():
	# 加载模型&测试数据
	model_mode = 'cascade'
	# model_mode = 'parallel'
	
	print('data loding ...')
	x_batch = data_loader()
	print('model loading ...')
	model = model_loader(model_mode)
	# 预测
	print('predicting...')
	y_test = model.predict(x_batch, batch_size=10)
	# 保存label
	print('result saving ...')
	for id in range(TEST_SIZE):
	    res = np.argmax(y_test[id], axis=-1)*60
	    cv2.imwrite('../data/test/label/img_{}.png'.format(id), res)
	print('done!')



if __name__ == '__main__':
	main()