# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: cnn3d.py

@time: 17-6-30 上午11:31

@desc: demo 拟用keras搭建3dcnn进行训练，输入正负样本估计在1:400

'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Convolution3D, MaxPool3D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# # Generate dummy data
# x_train = np.random.random((100, 10, 32, 32, 1))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# x_test = np.random.random((20, 10, 32, 32, 1))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
working_path = "/media/soffo/本地磁盘/tc/train/cubes/"
val_path = "/media/soffo/本地磁盘/tc/val/cubes/"
xneg = np.load(working_path+'neg/negLKDS-00335.npy')
yneg = np.zeros(xneg.shape[0])
xpos = np.load(working_path+'posAll.npy')
ypos = np.zeros(xpos.shape[0])

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv3D(64, (3, 3, 3), activation='relu', input_shape=(10, 32, 32, 1), padding='same'))
model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(MaxPool3D(pool_size=(2, 2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4),metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('/media/soffo/本地磁盘/tc/train/log/net3d.hdf5', monitor='val_loss', save_best_only=True)
model.fit(x_train, y_train, batch_size=1, epochs=50, verbose=1, shuffle=True,
          callbacks=[model_checkpoint], validation_data=[x_test, y_test])
