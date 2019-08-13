import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from collections import Counter
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

class DataLoader():
	def __init__(self, input_shape=(128, 128)):
		input_shape = input_shape	

	def load_data(self):
		dataset, info = tfds.load(name="voc2007", with_info=True)		
		class_names=info.features['labels'].names		
		
		X_train = []
		y_train = []
		for example in tfds.as_numpy(dataset['train']):
			new_img = example['image']
			new_img = cv.resize(new_img, input_shape[:2],interpolation = cv.INTER_AREA) 
			
			y = example['objects']['label']
			ids = example['objects']['is_difficult']
			y = y[~ids]
			
			if y.shape[0]<3:
				if y.shape[0]==1 or y[0]==y[1]:
					X_train.append(new_img)
					y_train.append(y[0])      
				else:
					for i in range(y.shape[0]):
						X_train.append(new_img)
						y_train.append(y[i])
			else:
				X_train.append(new_img)
				c = Counter(example['objects']['label'])
				y = c.most_common(1)[0][0]  
				y_train.append(y)
		train = (np.asarray(X_train),np.asarray(y_train))


		X_val = []
		y_val = []
		for example in tfds.as_numpy(dataset['validation']):
			new_img = example['image']
			new_img = cv.resize(new_img, input_shape[:2],interpolation = cv.INTER_AREA) 
			
			y = example['objects']['label']
			ids = example['objects']['is_difficult']
			y = y[~ids]
			
			if y.shape[0]<3:
				if y.shape[0]==1 or y[0]==y[1]:
					X_val.append(new_img)
					y_val.append(y[0])      
				else:
					for i in range(y.shape[0]):
						X_val.append(new_img)
						y_val.append(y[i])
			else:
				X_val.append(new_img)
				c = Counter(example['objects']['label'])
				y = c.most_common(1)[0][0]  
				y_val.append(y)
		val = (np.asarray(X_val),np.asarray(y_val))

		X_test = []
		y_test = []
		for example in tfds.as_numpy(dataset['test']):
			new_img = example['image']
			new_img = cv.resize(new_img, input_shape[:2],interpolation = cv.INTER_AREA) 
			
			y = example['objects']['label']
			ids = example['objects']['is_difficult']
			y = y[~ids]
			
			if y.shape[0]<3:
				if y.shape[0]==1 or y[0]==y[1]:
					X_test.append(new_img)
					y_test.append(y[0])      
				else:
					for i in range(y.shape[0]):
						X_test.append(new_img)
						y_test.append(y[i])
			else:
				X_test.append(new_img)
				c = Counter(example['objects']['label'])
				y = c.most_common(1)[0][0]  
				y_test.append(y)
		test = (np.asarray(X_test),np.asarray(y_test))

		del dataset
		return train, val, test, class_names
