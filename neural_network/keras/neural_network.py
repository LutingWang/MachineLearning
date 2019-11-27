#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:17:07 2019

@author: lutingwang
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

X_train = np.load('X_kannada_MNIST_train.npz')['arr_0']
y_train = np.load('y_kannada_MNIST_train.npz')['arr_0']
X_test = np.load('X_kannada_MNIST_test.npz')['arr_0']

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (5, 5), input_shape = (28, 28, 1), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units = 128, activation = 'relu'))
model.add(keras.layers.Dense(units = 10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, batch_size = 200, verbose = 1)
result = model.predict_classes(X_test)

f = open('submission.csv', 'w')
ans = input('save result? y/[n]: ')
if ans == 'y':
    f.write("id,label\n")
    for i in range(10000):
        f.write(str(i + 1) + ',' + str(result[i]) + '\n')
f.close()