# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: datagen.py

@time: 17-7-5 下午3:05

@desc: 
    1. 整合正反例数据
    2. 随机打乱
    3. 用于keras大数据迭代器
'''

from sklearn.utils import shuffle
import numpy as np
from glob import glob

# real data
train_path = "/media/soffo/本地磁盘/tc/train/cubes/"
val_path = "/media/soffo/本地磁盘/tc/val/cubes/"
cubesize = np.load('/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/feng/cube.npy')
cubexhalf = cubesize[0]
cubeyhalf = cubesize[1]
cubezhalf = cubesize[2]


def tainDataGen(working_path=''):
    print('-' * 50)
    print('Merging data ...')

    # xneg = np.r_[np.load(working_path + 'negbackup/merge/' + 'neg0.npy'), np.load(
    #     working_path + 'negbackup/merge/' + 'neg1.npy')]
    xneg = np.array([])
    xneg.shape = (0, 1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2)
    file_list = glob(working_path + "negbackup/merge/*.npy")
    for file in file_list:
        xnegnew = np.load(file)
        xneg = np.r_[xneg, xnegnew]
    yneg = np.zeros(xneg.shape[0]).astype(dtype=float)
    print(xneg.shape)
    xpos = np.r_[np.load(working_path + 'posbackup/' + 'posAll.npy'), np.load(
        working_path + 'posbackup/' + 'posxcubes.npy'), np.load(working_path + 'posbackup/' + 'posycubes.npy'), np.load(
        working_path + 'posbackup/' + 'posxycubes.npy')]
    ypos = np.ones(xpos.shape[0]).astype(dtype=float)
    print(xpos.shape)
    # double check cube数据是否是结点样式
    # for x in xpos:
    #     for xx in x[0]:
    #         plt.subplots()
    #         plt.imshow(xx)
    # for x in xneg:
    #     for xx in x[0]:
    #         plt.subplots()
    #         plt.imshow(xx)

    xneg = np.r_[xneg, xpos]
    yneg = np.r_[yneg, ypos]
    print('shuffling data ...')
    xneg, yneg = shuffle(xneg, yneg)
    print('Saving data ...')
    np.save(working_path + 'datagen/xtrain.npy', xneg)
    np.save(working_path + 'datagen/ytrain.npy', yneg)
    return 0

# def test


tainDataGen(working_path=val_path)
# tainDataGen(working_path=train_path)

