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
from tqdm import tqdm
path = '/home/soffo/Desktop/evaluationScript/evaluationScript/val/log/froc19/'
dup = pd.read_csv(path + 'submit.csv')
ids = set(dup['seriesuid'])

newdf = pd.DataFrame()

for theid in tqdm(ids):
    minipd = dup[dup['seriesuid'] == theid].copy()
    #    print minipd
    for i in range(len(minipd)):
        centroid1 = np.array([minipd.iloc[i]['coordX'], minipd.iloc[i]['coordY'], minipd.iloc[i]['coordZ']])
        probability1 = minipd.iloc[i]['probability']
        isuniq = True
        for j in range(i + 1, len(minipd)):
            centroid2 = np.array([minipd.iloc[j]['coordX'], minipd.iloc[j]['coordY'], minipd.iloc[j]['coordZ']])
            probability2 = minipd.iloc[j]['probability']
            if np.linalg.norm(centroid1 - centroid2) < 5:
                isuniq = False
                centroid = np.mean([centroid1, centroid2], axis=0)
                #                print "mean of center:",centroid
                # probability = np.max([probability1, probability2])
                probability = np.mean([probability1, probability2])
                minipd.iloc[j]['coordX'], minipd.iloc[j]['coordY'], minipd.iloc[j]['coordZ'] = centroid[0], centroid[1], \
                                                                                               centroid[2]
                minipd.iloc[j]['probability'] = probability
        if isuniq:
            newdf = newdf.append(minipd.iloc[i])

newdf.to_csv(path + 'uniq.csv', index=False, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
