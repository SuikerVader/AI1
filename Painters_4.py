# MNIST number classification.

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from skimage.transform import resize 
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout

import matplotlib.pylab as plt
import numpy as np
import pandas as pd 
import cv2

def main():

    ############# LOADING THE DATA
    #
    # Load de MNIST data set, je krijgt al ineens de train en test sets terug.
  
    print('-----------------------------------------------------------------')
    print('\nGetting painting images from folders.\n')
    from PIL import Image
    import glob
    image_list = []
    label_list = []
    training_image_list = []
    training_label_list = []
    test_image_list = []
    test_label_list = []
    counter = 0

    for filename in glob.glob('Paintings/Bruegel/*'):
        im = cv2.imread(filename)
        if im is None:
            print("Couldn't open file %s" % filename)
        else:
            if counter < 100:
                training_image_list.append(im)
                training_label_list.append(0)
                counter += 1
            else:
                test_image_list.append(im)
                test_label_list.append(0)

    counter = 0

    for filename in glob.glob('Paintings/Mondriaan/*'):
        im = cv2.imread(filename)
        if im is None:
            print("Couldn't open file %s" % filename)
        else:
            if counter < 100:
                training_image_list.append(im)
                training_label_list.append(1)
                counter += 1
            else:
                test_image_list.append(im)
                test_label_list.append(1)

    counter = 0

    for filename in glob.glob('Paintings/Picasso/*'):
        im = cv2.imread(filename)
        if im is None:
            print("Couldn't open file %s" % filename)
        else:
            if counter < 100:
                training_image_list.append(im)
                training_label_list.append(2)
                counter += 1
            else:
                test_image_list.append(im)
                test_label_list.append(2)

    counter = 0

    for filename in glob.glob('Paintings/Rubens/*'):
        im = cv2.imread(filename)
        if im is None:
            print("Couldn't open file %s" % filename)
        else:
            if counter < 100:
                training_image_list.append(im)
                training_label_list.append(3)
                counter += 1
            else:
                test_image_list.append(im)
                test_label_list.append(3)



    print('-----------------------------------------------------------------')
    print('\nLength of training data.\n')
    print(len(training_image_list))

    print('-----------------------------------------------------------------')
    print('\nShow Bruegel Painting.\n')
    imgplot = plt.imshow(training_image_list[0])
    plt.show()

    print('-----------------------------------------------------------------')
    print('\nShow Mondriaan Painting.\n')
    imgplot = plt.imshow(training_image_list[100])
    plt.show()

    print('-----------------------------------------------------------------')
    print('\nShow Picasso Painting.\n')
    imgplot = plt.imshow(training_image_list[200])
    plt.show()

    print('-----------------------------------------------------------------')
    print('\nShow Rubens Painting.\n')
    imgplot = plt.imshow(training_image_list[300])
    plt.show()

    instance1 = test_image_list[0]
    instance2 = test_image_list[100]


    ############# Reshaping the images
    #
    # Reshape de images tot een grootte van 28 op 28 pixels en zet de waarde van elke pixel om in floats van 0 tot 1.

    print('-----------------------------------------------------------------')
    print('\nPreprocessing.\n')
    training_image_list = np.array(training_image_list)
    print(training_image_list.shape)
    image = []
    for i in range(0, len(training_image_list)):
        a = resize(training_image_list[i], preserve_range=True, output_shape=(224,224)).astype(float)      # reshaping to 224*224*3
        image.append(a)
    training_image_list = np.array(image)

    image = []
    for i in range(0, len(test_image_list)):
        a = resize(test_image_list[i], preserve_range=True, output_shape=(224,224)).astype(float)      # reshaping to 224*224*3
        image.append(a)
    test_image_list = np.array(image)

    training_label_list = to_categorical(training_label_list)             # Hot encoding.
    test_label_list = to_categorical(test_label_list)             # Hot encoding.

    training_image_list = preprocess_input(training_image_list, mode='tf')      # preprocessing the input data

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    training_image_list = base_model.predict(training_image_list)
    test_image_list = base_model.predict(test_image_list) 

    training_image_list = training_image_list.reshape(400, 7*7*512)     # converting to 1-D
    test_image_list = test_image_list.reshape(155, 7*7*512)

    train = training_image_list/training_image_list.max()       # centering the data
    test_image_list = test_image_list/training_image_list.max()
    ############# MODELLING
    #
    # Build the model.

    print('-----------------------------------------------------------------')
    print('\nConfiguring the network.\n')

    model = Sequential()
    model.add(InputLayer((7*7*512,)))                       # input layer
    model.add(Dense(units=1024, activation='sigmoid'))      # hidden layer
    model.add(Dense(units=1024, activation='sigmoid'))      # hidden layer
    model.add(Dense(units=1024, activation='sigmoid'))      # hidden layer
    model.add(Dense(units=1024, activation='sigmoid'))      # hidden layer
    model.add(Dense(4, activation='softmax'))               # output layer

    print('-----------------------------------------------------------------')
    print('\nModel summary')
    print(model.summary())

    print('\n-----------------------------------------------------------------')
    print('\nCompiling the network.\n')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    ############# TRAINING
    #
    # Train the model.

    print('-----------------------------------------------------------------')
    print('\nTraining the network.\n')

    batch_size = 128  
    epochs = 50

    history = model.fit(training_image_list, 
                        training_label_list,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(test_image_list, test_label_list))

    ############# TESTING
    #
    # Evaluate the model on the test data.

    print('-----------------------------------------------------------------')
    print('\nTest the network.\n')

    test_loss, test_acc = model.evaluate(test_image_list, test_label_list)
    print('\nTest Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    ############# MAKE PREDICTIONS
    #

    print('-----------------------------------------------------------------')
    print('\nUsing the model, Making predictions.\n')

    # Get predicted labels.
    new_labels = model.predict_classes(test_image_list)

    '''
    print('\nPredictions:\n')

    for i in range(10):
      print("Target = %s, Predicted = %s" % (labels_test[i], new_labels[i]))
    '''

    # Show the inputs and predicted outputs.
    plt.imshow(instance1)
    plt.show()
    
    print("Predicted = %s" % new_labels[0])

    plt.imshow(instance2)
    plt.show()

    print("Predicted = %s" % new_labels[100])

main()