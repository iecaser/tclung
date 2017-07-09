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
from tqdm import tqdm

# working_path = "/media/soffo/本地磁盘/tc/val/cubes/negbackup/"
working_path = "/media/soffo/本地磁盘/tc/train/cubes/negbackup/"
cubesize = np.load('/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/feng/cube.npy')
cubexhalf = cubesize[0]
cubeyhalf = cubesize[1]
cubezhalf = cubesize[2]
sumsize = 200
cubexhalf = 16
cubeyhalf = 16
cubezhalf = 16
file_list = glob(working_path + "*.npy")
data = np.array([])
data.shape = (0, 1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2)
cnt = 0
i = 0
for file in tqdm(file_list):
    cnt += 1
    datatemp = np.load(file)
    # datatemp = datatemp.reshape(datatemp.shape[0], 1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2)
    data = np.r_[data, datatemp]
    if cnt == sumsize or i == len(file_list)-1:
        np.save(working_path + 'merge/' + 'neg{}.npy'.format(i // sumsize), data)
        data = np.array([])
        data.shape = (0, 1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2)
        cnt = 0
    i += 1
pass
