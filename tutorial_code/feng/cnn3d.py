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
train_path = "/media/soffo/本地磁盘/tc/train/"
val_path = "/media/soffo/本地磁盘/tc/val/"
test_path = "/media/soffo/本地磁盘/tc/test/"
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


# def loaddata():
#     print('-' * 30)
#     print('Loading data ...')
#     print('-' * 30)
#
#     xneg = np.load(working_path + 'negbackup0704-0.8/merge/' + 'neg0.npy')
#     yneg = np.zeros(xneg.shape[0]).astype(dtype=float)
#     xpos = np.load(working_path + 'posbackup/' + 'posAll.npy')
#     # xposall = np.load(working_path + 'posbackup/' + 'posAll.npy')
#     # xposx = np.load(working_path + 'posbackup/' + 'posxcubes.npy')
#     # xposy = np.load(working_path + 'posbackup/' + 'posycubes.npy')
#     # xposxy = np.load(working_path + 'posbackup/' + 'posxycubes.npy')
#     # xpos = np.r_[xposall, xposx]
#     ypos = np.ones(xpos.shape[0]).astype(dtype=float)
#
#     # double check cube数据是否是结点样式
#     # for x in xpos:
#     #     for xx in x[0]:
#     #         plt.subplots()
#     #         plt.imshow(xx)
#     # for x in xneg:
#     #     for xx in x[0]:
#     #         plt.subplots()
#     #         plt.imshow(xx)
#
#     xneg = np.r_[xneg, xpos]
#     yneg = np.r_[yneg, ypos]
#     xTrain, xTest, yTrain, yTest = train_test_split(xneg, yneg)
#     xTrainmean = np.mean(xTrain)
#     xTrainstd = np.std(xTrain)
#     xTrain -= xTrainmean
#     xTrain /= xTrainstd
#     xTest -= xTrainmean
#     xTest /= xTrainstd
#     return xTrain, yTrain, xTest, yTest


def cnntrain(use_existing=False):

    # xtrain = np.load('/media/soffo/本地磁盘/tc/train/cubes/datagen/xtrain.npy')
    # ytrain = np.load('/media/soffo/本地磁盘/tc/train/cubes/datagen/ytrain.npy')
    # 一定要做归一化!!
    #     xTrainmean = np.mean(xTrain)
    #     xTrainstd = np.std(xTrain)
    #     xTrain -= xTrainmean
    #     xTrain /= xTrainstd
    #     xTest -= xTrainmean
    #     xTest /= xTrainstd
    xval = np.load('/media/soffo/本地磁盘/tc/val/cubes/datagen/xval.npy')
    yval = np.load('/media/soffo/本地磁盘/tc/val/cubes/datagen/yval.npy')
    # val也要用train的归一化
    model_checkpoint = ModelCheckpoint('/media/soffo/本地磁盘/tc/train/log/net3d.hdf5', monitor='val_loss',
                                       save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    model = get3dcnn()
    if use_existing:
        model.load_weights('/media/soffo/本地磁盘/tc/train/log/net3d.hdf5')
    # model.fit(xtrain, ytrain, batch_size=3, epochs=50, verbose=1, shuffle=True,
    model.fit(xval, yval, batch_size=3, epochs=50, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, earlystop], validation_data=[xval, yval])
    return model


def cnnpredict(working_path=test_path):
    print('-' * 50)
    print('Predicting ...')
    print('-' * 50)
    cubexhalf = 16
    cubeyhalf = 16
    cubezhalf = 16
    file_list = glob(working_path +'cubes/neg/*.npy')
    # file_list = glob(val_path +'posbackup/'+ "*.npy")
    model = get3dcnn()
    model.load_weights('/media/soffo/本地磁盘/tc/train/log/net3d.hdf5')
    for i, cubefile in enumerate(file_list):
        cubes = np.load(cubefile)
        cubemean = np.mean(cubes)
        cubestd = np.std(cubes)
        cubes -= cubemean
        cubes /= cubestd
        filename = re.split('\.|/', cubefile)[-2][3:]
        # 按照cubes顺序得到的coef
        # coornpy和dianpy文件也是按照这个顺序
        coef = model.predict(cubes)
        coef = coef.flatten()
        # 默认argsort是升序,取负保证index对于+coef而言是降序
        index = np.argsort(-coef)
        coef = coef[index]
        coor = np.load(working_path+'coor/coor{}.npy'.format(filename))
        coor = coor[index]
        dia = np.load(working_path+'dia/dia{}.npy'.format(filename))
        dia = dia[index]
        # np.save(working_path+'coef/coef{}.npy'.format(filename), coef)

        # debug
        print(filename)
        print('-'*50)
        print('coef:')
        print(coef[:10])
        print('coor:')
        print(coor[:10])
        print('dia')
        print(dia[:10])


# load data
# xTrain, yTrain, xTest, yTest = loaddata()
# train
model = cnntrain(use_existing=False)
# predict
cnnpredict(working_path=val_path)
