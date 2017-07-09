# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: zcj.py

@time: 17-7-8 上午11:04

@desc:

'''
import pandas as pd
import numpy as np

path = '/home/soffo/Desktop/evaluationScript/evaluationScript/val/log/backup/'
cj = pd.read_csv(path + 'val_submit3 (copy).csv')
cjj = cj.copy()
xf = pd.read_csv(path + 'submit.csv')
cjids = set(list(cj['seriesuid']))
xfids = set(list(xf['seriesuid']))
# # 下面函数用于补缺
# for xfid in xfids:
#     if xfid not in cjids:
#         minipd = xf[xf['seriesuid'] == xfid]
#         cjj = pd.concat([cjj, minipd])
#     else:
#         print(xfid)
# cjj.to_csv(path + 'sum.csv')

# 下面进行查重
newdf = pd.DataFrame()
uids = []
ps = np.array([])
coors = np.array([]).reshape(0,3)
for value in xf.values:
    uid = value[0]
    coor = np.array(value[1:4])
    p = np.array(value[-1])
    for uniqueUid in set(uids):


