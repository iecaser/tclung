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
import pandas as pd
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

def get3dcnn01():
    model = Sequential()

    model.add(Conv3D(64, (3, 3, 3), activation='relu', input_shape=(1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2),
                     padding='same'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='valid'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='valid'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='valid'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1.0e-5), metrics=['binary_accuracy'])
    return model
def get3dcnn014():
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=(1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2),
                     padding='same'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1.0e-5), metrics=['binary_accuracy'])
    return model


def get3dcnnmy():
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3), activation='relu', input_shape=(1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2),
                     padding='same'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='valid'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='valid'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='valid'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='valid'))
    # model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='valid'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='valid'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
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
    print('loading val data...')
    xval = np.load('/media/soffo/本地磁盘/tc/val/cubes/datagen/xval.npy')
    yval = np.load('/media/soffo/本地磁盘/tc/val/cubes/datagen/yval.npy')
    print('normalizing...')
    xval -= np.mean(xval)
    xval /= np.std(xval)
    # val也要用train的归一化
    print('splitting data...')
    xtrain, xval, ytrain, yval = train_test_split(xval, yval)
    model_checkpoint = ModelCheckpoint('./net3d.hdf5', monitor='val_loss',
                                       save_best_only=True)
    print('getting model...')
    model = get3dcnn014()
    if use_existing:
        print('loading model weights...')
        model.load_weights('./net3d.hdf5')
    print('fitting...')
    model.fit(xval, yval, batch_size=3, epochs=50, verbose=1, shuffle=True,
              callbacks=[model_checkpoint], validation_data=[xval, yval])
    return model


def cnnpredict(working_path=''):
    print('-' * 50)
    print('Predicting ...')
    print('-' * 50)
    cubexhalf = 16
    cubeyhalf = 16
    cubezhalf = 16
    # file_list = glob(working_path + 'cubes/neg/*.npy')
    file_list = glob(working_path + 'cubes/negbackup/*.npy')
    # file_list = glob(val_path +'posbackup/'+ "*.npy")
    trainmean, trainstd = np.load(working_path + 'meanstd.npy')
    # model = get3dcnn1()
    # model.load_weights('/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/feng/net3d.hdf5')
    model = get3dcnn01()
    model.load_weights('/media/soffo/本地磁盘/tc/train/log/net3d01.hdf5')

    seriesuids = []
    coors = np.array([]).reshape(0, 3)
    probabilitys = np.array([]).reshape(0, )
    dias = np.array([]).reshape(0, )
    man = []
    for i, cubefile in enumerate(file_list):
        cubes = np.load(cubefile)
        # cubemean = np.mean(cubes)
        # cubestd = np.std(cubes)
        cubes -= trainmean
        cubes /= trainstd
        seriesuid = re.split('\.|/', cubefile)[-2][3:]
        # 按照cubes顺序得到的probability
        # coornpy和dianpy文件也是按照这个顺序
        probability = model.predict(cubes)
        probability = probability.flatten()-0.1
        # 默认argsort是升序,取负保证index对于+probability而言是降序
        index = np.argsort(-probability)
        probability = probability[index]
        coor = np.load(working_path + 'coor/coor{}.npy'.format(seriesuid))
        coor = coor[index]
        dia = np.load(working_path + 'dia/dia{}.npy'.format(seriesuid))
        dia = dia[index]
        # 依据简单规则进行粗筛选,后续和unet在csv-level结合
        # 后续或可根据rio训练二级分类器,边界信息一定要用
        probability = probability[:8]
        coor = coor[:8]
        dia = dia[:8]
        aFilter = probability > 0.7
        cnt = np.sum(aFilter)
        if cnt == 0:
            man.append(seriesuid)
            print('#' * 30 + '\t\t' + seriesuid + '\t\t未检出' + '\t\t\t' + '#' * 30)
            continue
        probability = probability[aFilter]
        coor = coor[aFilter]
        dia = dia[aFilter]
        # submit format
        seriesuids.extend([seriesuid] * cnt)
        probabilitys = np.r_[probabilitys, probability]
        coors = np.r_[coors, coor]
        dias = np.r_[dias, dia]

        # debug
        print('on\t\t'+seriesuid+'\t\t'+'-' * 20)
        print('probability:')
        print(probability)
        print('coor:')
        print(coor)
        print('dia')
        print(dia)
    # submit = pd.DataFrame(columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'dia'])
    submit = pd.DataFrame(columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
    submit['seriesuid'] = seriesuids
    submit[['coordX', 'coordY', 'coordZ']] = coors
    submit['probability'] = probabilitys
    # submit['dia'] = dias
    man = np.array(man)
    print('需要手动查找个数:{}'.format(len(man)))
    print(man)
    np.save(working_path+'log/man.npy', man)
    submit.to_csv(working_path+'log/submit.csv', index=False)


# load data
# xTrain, yTrain, xTest, yTest = loaddata()
# train
# model = cnntrain(use_existing=False)
# model = cnntrain(use_existing=True)
# predict
cnnpredict(working_path=test_path)
# cnnpredict(working_path=val_path)
