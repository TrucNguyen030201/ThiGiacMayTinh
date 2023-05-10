import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

path = ""
# categories = ['Figs']
categories ='Figs','MacMat','NhuaRuoi', 'Nightshade', 'Physalis', 'Pokeberri', 'SimRung', 'Snakefruit', 'Syzygium', 'ThanhMai', 'TramRung', 'TyBa'


# 1.5 
data = []
labels = []
imagePaths = []

HEIGHT = 128 #
WIDTH = 128 #
N_CHANNELS = 3

for k, category in enumerate(categories):
    for f in os.listdir(path + category):
        imagePaths.append([path + category+'/'+f, k])

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))
    data.append(image)

    label = imagePath[1]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

EPOCHS = 25
# INIT_LR = 1e-3
BS = 32

trainY = np_utils.to_categorical(trainY, 12)

model = Sequential()
model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1)