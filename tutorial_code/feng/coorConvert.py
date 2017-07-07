# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: coorConvert.py

@time: 17-7-7 下午4:00

@desc:

'''

import pandas as pd
import numpy as np
import SimpleITK as sitk

man = pd.read_csv('/home/soffo/Desktop/man.csv')
ids = man['id']
xyzs = np.array(man[['x', 'y', 'z']])
for i, id_ in enumerate(ids):
    img_file = '/media/soffo/本地磁盘/tc/test/data/{}.mhd'.format(id_)
    itk_img = sitk.ReadImage(img_file)
    origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
    xyzs[i] = xyzs[i] * spacing + origin
man[['x', 'y', 'z']] = xyzs
man.to_csv('/home/soffo/Desktop/newman.csv')
pass
