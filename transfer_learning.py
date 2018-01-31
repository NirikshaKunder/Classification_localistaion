from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format
import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm

import xml.etree.ElementTree


import glob
path_xml = glob.glob("details/*.xml")

def get_name(a):
	e = xml.etree.ElementTree.parse(a).getroot()
	result = []
	image_path = ''
	

	for i in e:
		if i.tag == 'filename':
			image_path = i.text
		if i.tag == 'object':
			for j in i:
				if(j.tag == 'name'):
					result.append(j.text)
	#print a, image_path, result
	return result, image_path


IMG_SIZE = 28
def read_img(img_path):
	img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	img = np.array(img)
	return img

# take a sample image
img = read_img('details_image/2007_000032.jpg')
data_set = []
class_id = dict()
id_class = []
count = 0
sample = 5000
for i in path_xml:
	tmp = get_name(i)
	for j in tmp[0]:
		if(j not in class_id):
			class_id[j] = count
			count += 1
			id_class.append(j)
	tt = [class_id[i] for i in tmp[0]]
	data_set.append((read_img('details_image/' + tmp[1]), tt))
	sample -= 1
	if(sample == 0):
		break

print 'number of classes: ', len(id_class)
num_classes = len(id_class)

print img
print len(img), len(img[0]), len(img[0][0])
train_X = [data_set[i][0] for i in range(len(data_set))]
train_X = np.array(train_X, np.float32) / 255
print 'tetette ' ,train_X.shape
train_label = []
for i in range(len(data_set)):
	train_label.append(list(range(num_classes)))
	for j in data_set[i][1]:
		train_label[i][j] = 1
train_label = np.array(train_label)
input = Input(shape=(3,200,200),name = 'train_X')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(20, activation='softmax', name='predictions')(x)

my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1)

y = model.predict(train_X[:10])

