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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from glob import glob
import numpy as np
import re
# real data
working_path = "/media/soffo/本地磁盘/tc/train/cubes/"
val_path = "/media/soffo/本地磁盘/tc/val/cubes/"
cubexhalf = 16
cubeyhalf = 16
cubezhalf = 16


def get3dcnn():
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=(1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2),
                     padding='same'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1.0e-5), metrics=['binary_accuracy'])
    return model


def cnntrain(use_existing=False):
    print('-' * 30)
    print('Loading data ...')
    print('-' * 30)
    xneg = np.load(working_path + 'negbackup/merge/' + 'neg0.npy')
    # xneg0 = np.load(working_path + 'negbackup/merge/' + 'neg0.npy')
    # xneg1 = np.load(working_path + 'negbackup/merge/' + 'neg1.npy')
    # xneg = np.r_[xneg0, xneg1]
    # yneg = np.zeros(xneg.shape[0]).astype(dtype=float)
    yneg = np.zeros(xneg.shape[0]).astype(dtype=float)
    # test 阶段控制1：20正负样本
    xposall = np.load(working_path + 'posbackup/' + 'posAll.npy')
    xposx = np.load(working_path + 'posbackup/' + 'posxcubes.npy')
    # xposy = np.load(working_path + 'posbackup/' + 'posycubes.npy')
    # xposxy = np.load(working_path + 'posbackup/' + 'posxycubes.npy')
    xpos = np.r_[xposall, xposx]
    ypos = np.ones(xpos.shape[0]).astype(dtype=float)

    # double check cube数据是否是结点样式
    # for x in xpos:
    #     for xx in x[0]:
    #         plt.subplots()
    #         plt.imshow(xx)
    # for x in xneg:
    #     for xx in x[0]:
    #         plt.subplots()
    #         plt.imshow(xx)

    datax = np.r_[xneg, xpos]
    datay = np.r_[yneg, ypos]
    xTrain, xTest, yTrain, yTest = train_test_split(datax, datay)
    # xTrainmean = np.mean(xTrain)
    # xTrainstd = np.std(xTrain)
    # xTrain -= xTrainmean
    # xTrain /= xTrainstd
    # xTest -= xTrainmean
    # xTest /= xTrainstd

    # n = 100
    # m = 30
    # xTrain = np.random.rand(n, 1, 32, 32, 32)
    # yTrain = np.random.randint(0, 2, n)
    # xTest = np.random.rand(m, 1, 32, 32, 32)
    # yTest = np.random.randint(0, 2, m)



    model_checkpoint = ModelCheckpoint('/media/soffo/本地磁盘/tc/train/log/net3d.hdf5', monitor='val_loss',
                                       save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    model = get3dcnn()
    if use_existing:
        model.load_weights('/media/soffo/本地磁盘/tc/train/log/net3d.hdf5')
    model.fit(xTrain, yTrain, batch_size=3, epochs=50, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, earlystop], validation_data=[xTest, yTest])
    return model




def cnnpredict():
    print('-' * 30)
    print('Predicting ...')
    print('-' * 30)
    cubexhalf = 16
    cubeyhalf = 16
    cubezhalf = 16
    file_list = glob(val_path + "*.npy")
    model = get3dcnn()
    model.load_weights('/media/soffo/本地磁盘/tc/train/log/net3d.hdf5')
    for i, cubefile in enumerate(file_list):
        cubes = np.load(cubefile)
        filename = re.split('\.|/', cubefile)[-2][3:]
        coef = model.predict(cubes)
        np.save('./{}coef.npy'.format(filename), coef)
        print(cubefile)
        print(coef)


# cnntrain(use_existing=True)
model = cnntrain(use_existing=False)
