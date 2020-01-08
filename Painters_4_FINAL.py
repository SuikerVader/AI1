import tensorflow as tf
import keras

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import glob
import os.path

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

from sklearn.externals import joblib 

training_image_list = []
training_label_list = []
validation_image_list = []
validation_label_list = []
test_image_list = []
history = None

imageSize = 150
training_percent = 0.7
validation_percent = 0.15
test_percent = 0.15

painters = ["RubensAll", "PicassoAll", "MondriaanAll", "BruegelAll"]

model = models.Sequential()

def random_rotation(image):
	random_degree = random.uniform(-25, 25)
	return sk.transform.rotate(image, random_degree)

def random_noise(image):
	return sk.util.random_noise(image)

def horizontal_flip(image):
	return image[:, ::-1]

def plot_graphs():
	global history
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()


def fillArray(painter: painters, i: int):
	print('-----------------------------------------------------------------')
	print('\nGetting painting images from folder.' + painter + '\n')
	counter = 0
	totalImages = len(glob.glob('Paintings/' + painter +'/*'))
	for filename in glob.glob('Paintings/' + painter + '/*'):
		im = cv2.imread(filename)
		if im is None:
			print("Couldn't open file %s" % filename)
		else:
			if counter < totalImages * training_percent:
				training_image_list.append(cv2.resize(im, (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
				training_image_list.append(cv2.resize(random_rotation(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
				training_image_list.append(cv2.resize(random_noise(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
				training_image_list.append(cv2.resize(horizontal_flip(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
				for x in range(4):
                    training_label_list.append(i)
				counter += 1
			elif counter < totalImages * (training_percent + validation_percent):
				validation_image_list.append(cv2.resize(im, (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
				validation_image_list.append(cv2.resize(random_rotation(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
				validation_image_list.append(cv2.resize(random_noise(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
				validation_image_list.append(cv2.resize(horizontal_flip(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
				for x in range(4):
                    validation_label_list.append(i)
				counter += 1
			elif counter < totalImages * (training_percent + validation_percent + test_percent):
				test_image_list.append(cv2.resize(im, (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))


def create_model():
	print('\n==================================================================')
	print('MAKING THE MODEL.')

	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imageSize, imageSize, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))  # To avoid overfitting.
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(4, activation='softmax'))


def compile_model():
	print('COMPILING THE MODEL.')
	model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])


def train_model():
	print('-----------------------------------------------------------------')
	print('\nTraining the network.\n')

	batch_size = 25
	epochs = 250

	global history

	history = model.fit(training_image_list,
						training_label_list,
						batch_size=batch_size,
						epochs=epochs,
						verbose=2,
						validation_data=(validation_image_list, validation_label_list))

	joblib.dump(model, 'saved_model_4_painters.pkl') 


def test_model():
	############# TESTING
	#
	# Evaluate the model on the test data.

	print('-----------------------------------------------------------------')
	print('\nTest the network.\n')

	test_loss, test_acc = model.evaluate(validation_image_list, validation_label_list)

	model.summary()
	print('\nTest Loss:', test_loss)
	print('Test Accuracy:', test_acc)

	print('-----------------------------------------------------------------')
	print('\nUsing the model, Making predictions.\n')

	# Get predicted labels.
	new_labels = model.predict_classes(test_image_list)

	

	print(new_labels)

def predict_painting(model):
	global painters
	total = 0
	right = 0
	user_input = ("Geef de naam van een schilderij om te raden (Typ stop om te stoppen): ")
	while True:
		found = False
		user_input = input("Geef de naam van een schilderij om te raden (Typ stop om te stoppen): ")
		if user_input.lower() == "stop":
			if total != 0:
				print("Juist geraden: " + str((right/total)*100) + "%")
			break;
		for painter in painters:
			prefix = 'Paintings/' + painter + '\\'
			for filename in glob.glob(prefix + '*'):
				name_painting = filename.replace(prefix, '').replace('.jpg','').replace('.jpeg','').replace('.png','')
				if name_painting.lower() == user_input.lower():
					found = True
					im = cv2.imread(filename)
					imgplot = plt.imshow(im)
					plt.show()
					im = cv2.resize(im, (imageSize, imageSize), interpolation=cv2.INTER_CUBIC)
					img_tensor = np.expand_dims(im, axis=0)
					prediction = model.predict_classes(img_tensor)
					total += 1
					if prediction == 0:
						print("Het model is " + str(round(model.predict(img_tensor)[0][0] * 100)) + "% zeker dat het schilderij van Rubens is.")
					elif prediction == 1:
						print("Het model is " + str(round(model.predict(img_tensor)[0][1] * 100)) + "% zeker dat het schilderij van Picasso is.")
					elif prediction == 2:
						print("Het model is " + str(round(model.predict(img_tensor)[0][2] * 100)) + "% zeker dat het schilderij van Mondriaan is.")
					else:
						print("Het model is " + str(round(model.predict(img_tensor)[0][3] * 100)) + "% zeker dat het schilderij van Bruegel is.")
					verificatie = ""
					while True:
						verificatie = input("Heeft het model het juist geraden? (typ ja of nee)")
						if verificatie.lower() == "ja":
							right += 1
							print("Juist geraden: " + str((right/total)*100) + "%")
							break
						elif verificatie.lower() == "nee":
							print("Juist geraden: " + str((right/total)*100) + "%")
							break
					
		if not found:
			print("Geen schilderij met naam " + user_input + " gevonden. Probeer het nog een keer!")


def main():
	global model
	if os.path.isfile('saved_model_4_painters.pkl'):
		model = joblib.load('saved_model_4_painters.pkl')
	else:
		for i in range(len(painters)):
			fillArray(painters[i], i)
		global training_image_list
		global validation_image_list
		global test_image_list
		global training_label_list
		global validation_label_list
		training_image_list = np.array(training_image_list)
		validation_image_list = np.array(validation_image_list)
		test_image_list = np.array(test_image_list)
		training_label_list = to_categorical(training_label_list)
		validation_label_list = to_categorical(validation_label_list)
		create_model()
		compile_model()
		train_model()
		plot_graphs()
		test_model()
	predict_painting(model)
	

if __name__ == "__main__":
	main()
