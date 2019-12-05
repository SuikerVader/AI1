import tensorflow as tf
import keras

from keras import layers
from keras import models
from keras import optimizers

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import glob
from keras.utils import to_categorical

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

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

test_painting1 = 0
test_painting2 = 20

painters = ["RubensAll", "PicassoAll", "MondriaanAll"]

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
    print('\nGetting painting images from folders.\n')
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
                training_label_list.append(i)
                training_label_list.append(i)
                training_label_list.append(i)
                training_label_list.append(i)
                counter += 1
            elif counter < totalImages * (training_percent + validation_percent):
                validation_image_list.append(cv2.resize(im, (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
                validation_image_list.append(cv2.resize(random_rotation(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
                validation_image_list.append(cv2.resize(random_noise(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
                validation_image_list.append(cv2.resize(horizontal_flip(im), (imageSize, imageSize), interpolation=cv2.INTER_CUBIC))
                validation_label_list.append(i)
                validation_label_list.append(i)
                validation_label_list.append(i)
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
    model.add(layers.Dense(3, activation='softmax'))

def compile_model():
    print('COMPILING THE MODEL.')
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

def train_model():
    print('-----------------------------------------------------------------')
    print('\nTraining the network.\n')

    batch_size = 25
    epochs = 100

    global history

    history = model.fit(training_image_list,
                        training_label_list,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(validation_image_list, validation_label_list))


def test_model(instance1, instance2):
	############# TESTING
    #
    # Evaluate the model on the test data.

    print('-----------------------------------------------------------------')
    print('\nTest the network.\n')

    test_loss, test_acc = model.evaluate(validation_image_list, validation_label_list)
    print('\nTest Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    print('-----------------------------------------------------------------')
    print('\nUsing the model, Making predictions.\n')

    # Get predicted labels.
    new_labels = model.predict_classes(test_image_list)

    # Show the inputs and predicted outputs.
    plt.imshow(instance1)
    plt.show()

    print("Predicted = %s" % new_labels[test_painting1])

    plt.imshow(instance2)
    plt.show()

    print("Predicted = %s" % new_labels[test_painting2])



def main():

    for i in range(len(painters)):
        fillArray(painters[i], i)

    global training_image_list
    global validation_image_list
    global test_image_list
    global training_label_list
    global validation_label_list

    training_image_list = np.array((training_image_list))
    validation_image_list = np.array(validation_image_list)
    test_image_list = np.array(test_image_list)

    training_label_list = to_categorical(training_label_list)
    validation_label_list = to_categorical(validation_label_list)
        
    instance1 = test_image_list[test_painting1]
    instance2 = test_image_list[test_painting2]

    create_model()
    compile_model()
    train_model()
    test_model(instance1, instance2)
    plot_graphs()
    

if __name__ == "__main__":
    main()
