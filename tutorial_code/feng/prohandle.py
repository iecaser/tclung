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

mode = 1
# mode = 2
test_path = '/media/soffo/本地磁盘/tc/test/'
# filename = 'Untitled 1.csv'
filename = 'fakesubmit (copy).csv'
man = pd.read_csv('/home/soffo/Desktop/' + filename)
ids = man['seriesuid']
ids = set(list(ids))
p = man['probability']
print('ids count:', len(set(ids)))
# p -= 0.1
# man['probability'] = p
# man.to_csv('/home/soffo/Desktop/fake' + filename, index=False)
file_list = glob(test_path + 'data/*.mhd')
for file in file_list:
    seriesuid = re.split('\.|/', file)[-2]
    if seriesuid not in ids:
        print(seriesuid)
