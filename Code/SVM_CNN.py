import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from skimage.io import imread, imshow
from skimage.transform import resize
from tensorflow.keras.applications import InceptionV3, MobileNet, VGG16, DenseNet121, EfficientNetB7
from sklearn.svm import SVC
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

path="./Images/"
categories = ['Figs', 'MacMat', 'NhuaRuoi', 'Nightshade', 'Physalis', 'Pokeberri', 'SimRung', 'Snakefruit', 'Syzygium', 'ThanhMai', 'TramRung', 'TyBa']

data = []
labels = []
imagePaths = []
HEIGHT = 64
WIDTH = 64
N_CHANNELS = 3

EPOCHS = 5
INIT_LR = 1e-3
BS = 15

for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k])

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

for imagePath in imagePaths:
    image = imread(imagePath[0])
    image = resize(image, (WIDTH, HEIGHT), anti_aliasing=True)
    data.append(image)
    label = imagePath[1]
    labels.append(label)

data = np.array(data, dtype="float32") 
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

trainY = to_categorical(trainY, len(categories))

base_model = MobileNet(input_shape =(WIDTH, HEIGHT, N_CHANNELS), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
preds = Dense(len(categories), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds) 

for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-4:]:
    layer.trainable=True

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BS, verbose = 1, validation_data = (testX, to_categorical(testY, len(categories))))
end = time.time()
model.save("SVN_CNN.h5")

new_model=Model(inputs=model.input,outputs=model.get_layer('dense').output)
print("new_model:  ", new_model)

feat_train = new_model.predict(trainX)
print("feat_train: ", feat_train.shape)

feat_test = new_model.predict(testX)
print("feat_test: ", feat_test.shape)

# Initializing SVR model with rbf kernel
from sklearn.svm import SVR
print("Initializing SVR model...")
model_SVR = SVR(kernel="rbf", C=10000, gamma=0.1, epsilon=0.1, verbose=2)
model_SVR.fit(feat_train, np.argmax(trainY,axis=1))

pred = model_SVC.predict(feat_test)

predictions = model.predict(testX)
y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(testY, y_pred)
precision, recall, fscore, support = precision_recall_fscore_support(testY, y_pred, average='weighted')

# print the results
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision: %.2f" % precision)
print("Recall: %.2f" % recall)
print("F1-Score: %.2f" % fscore)
print("Support: ", support)
print("Time taken: ", end-start, " seconds")
