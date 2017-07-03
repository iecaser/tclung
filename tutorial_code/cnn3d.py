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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# # Generate dummy data
# xTrain = np.random.random((100, 1, 10, 32, 32))
# yTrain = np.random.randint(2, size=(100, 1))
# xTest = np.random.random((20, 1, 10, 32, 32))
# yTest = np.random.randint(2, size=(20, 1))

# real data
working_path = "/media/soffo/本地磁盘/tc/train/cubes/"
val_path = "/media/soffo/本地磁盘/tc/val/cubes/"
cubexhalf = 16
cubeyhalf = 16
cubezhalf = 5


xneg = np.load(working_path + 'negbackup/merge/' + 'neg0.npy')
yneg = np.zeros(xneg.shape[0])
# test 阶段控制1：20正负样本
xpos = np.load(working_path + 'posAll.npy')[:200]
ypos = np.ones(xpos.shape[0])

for x in xpos:
    for xx in x[0]:
        plt.subplots()
        plt.imshow(xx)

for x in xneg:
    for xx in x[0]:
        plt.subplots()
        plt.imshow(xx)



datax = np.r_[xneg, xpos]
datay = np.r_[yneg, ypos]
xTrain, xTest, yTrain, yTest = train_test_split(datax, datay)
xTrainmean = np.mean(xTrain)
xTrainstd = np.std(xTrain)
xTrain -= xTrainmean
xTrain /= xTrainstd
xTest -= xTrainmean
xTest /= xTrainstd

model = Sequential()
model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=(1, cubezhalf*2, cubexhalf*2, cubeyhalf*2), padding='same'))
model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(MaxPool3D(pool_size=(2, 2, 2)))
# model.add(Dropout(0.25))
model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
model.add(MaxPool3D(pool_size=(2, 2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1.0e-5), metrics=['binary_accuracy'])

model_checkpoint = ModelCheckpoint('/media/soffo/本地磁盘/tc/train/log/net3d.hdf5', monitor='val_loss',
                                   save_best_only=True)
model.fit(xTrain, yTrain, batch_size=2, epochs=30, verbose=1, shuffle=True,
          callbacks=[model_checkpoint], validation_data=[xTest, yTest])
