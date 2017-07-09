# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: prohandle.py

@time: 17-7-7 下午10:03

@desc:

'''

import pandas as pd
import numpy as np
import SimpleITK as sitk
from glob import glob
import re

test_path = '/media/soffo/本地磁盘/tc/test/'
# test_path = '/media/soffo/本地磁盘/tc/val/'
# filename = 'Untitled 1.csv'
# filename = 'submit0708.csv'
# filename = 'submitval.csv'
filename = 'cjsubmitnew.csv'
man = pd.read_csv('/home/soffo/Desktop/' + filename)
ids = man['seriesuid']
ids = set(list(ids))
print('ids count:', len(set(ids)))
p = man['probability']
p -= 0.2
# p *= 1.2
# p[p > 1] = 1.0
man['probability'] = p
man.to_csv('/home/soffo/Desktop/fake' + filename, index=False)
file_list = glob(test_path + 'data/*.mhd')
for file in file_list:
    seriesuid = re.split('\.|/', file)[-2]
    if seriesuid not in ids:
        print(seriesuid)
