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

path=""
categories = ['ChayRung', 'DuDeTrau', 'Figs','MacMat','NhuaRuoi', 'Nightshade', 'Physalis', 'Pokeberri', 'SimRung', 'Snakefruit', 'Syzygium', 'TaoMeo', 'ThanhMai', 'TramRung', 'TyBa']

# for category in categories:
#     fig, _ = plt.subplots(3,4)
#     fig.suptitle(category)
#     for k, v in enumerate(os.listdir(path+category)[:12]):
#         img = plt.imread(path+category+'/'+v)
#         plt.subplot(3, 4, k+1)
#         plt.axis('off')
#         plt.imshow(img)
#     plt.show()

# shape0 = []
# shape1 = []

# for category in categories:
#     for files in os.listdir(path+category):
#         shape0.append(plt.imread(path+category+'/'+ files).shape[0])
#         shape1.append(plt.imread(path+category+'/'+ files).shape[1])
#     print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
#     print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
#     shape0 = []
#     shape1 = []

data = []#dữ liệu
labels = []#nhãn
imagePaths = []
HEIGHT = 128
WIDTH = 128

N_CHANNELS = 3

for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) 

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    # print(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    label = imagePath[1]
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

plt.subplots(3,13)
for i in range(13):
    plt.subplot(3,13, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[labels[i]])
# plt.show()

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=30)

trainY = np_utils.to_categorical(trainY, 13)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

EPOCHS = 20
INIT_LR = 1e-3
BS = 16

class_names = categories

from tensorflow import keras
import keras.layers as layers
model = keras.Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(11, 11),
                        strides=(4, 4), activation="relu",
                        input_shape=(WIDTH, HEIGHT, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                        strides=(1, 1), activation="relu",
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(1, 1), activation="relu",
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(1, 1), activation="relu",
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),
                        strides=(1, 1), activation="relu",
                        padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(class_names), activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1)



# đoạn này dùng để save mô hình 
model.save("lenet.h5")
model.save_weight('lenet_weight.h5')
# đoạn này dùng để kiểm tra mô hình

from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score

pred = model.predict(testX)
predictions = argmax(pred, axis=1) # return to label

cm = confusion_matrix(testY, predictions)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Model confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + categories)
ax.set_yticklabels([''] + categories)

# for i in range(5):
#     for j in range(5):
#         ax.text(i, j, cm[j, i], va='center', ha='center')

plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()

from sklearn.metrics import precision_recall_fscore_support as score
accuracy = accuracy_score(testY, predictions)
print("Accuracy : %.2f%%" % (accuracy*100.0))
precision, recall, fscore, support = score(testY, predictions)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

