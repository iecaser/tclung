# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: negcubes.py

@time: 17-7-2 上午9:05

@desc:

'''

from glob import glob
import numpy as np

working_path = "/media/soffo/本地磁盘/tc/train/cubes/negbackup/"
file_list = glob(working_path + "*.npy")
data = np.array([])
data.shape = (0, 1, 10, 32, 32)
cnt = 0
for i, file in enumerate(file_list):
    cnt += 1
    datatemp = np.load(file)
    datatemp = datatemp.reshape(datatemp.shape[0], 1, 10, 32, 32)
    data = np.r_[data, datatemp]
    if cnt == 20 or i == len(file_list) - 1:
        np.save(working_path + 'merge/' + 'neg{}.npy'.format(i // 20), data)
        data = np.array([])
        data.shape = (0, 1, 10, 32, 32)
        cnt = 0

pass
