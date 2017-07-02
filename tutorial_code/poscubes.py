# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: poscubes.py

@time: 17-7-1 上午9:53

@desc:

'''

import re

# from tutorial_code.lung_segmentation import *
try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage.filters import gaussian, median
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure, morphology

############
# 全局

luna_path = r"/media/soffo/本地磁盘/tc/val/"
# luna_path = r"/media/soffo/本地磁盘/tc/train/"

luna_subset_path = luna_path + 'data/'
output_path = luna_path + 'tutorial/'
file_list = glob(luna_subset_path + "*.mhd")


# ------------------------------------------------------

def extractNodeCenter(mini_df, ifprint=False):
    # 每个node标记出位置，marker选用其他
    nodesCenter = []
    for node_idx, cur_row in mini_df.iterrows():
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        # diam = cur_row["diameter_mm"]
        center = np.array([node_x, node_y, node_z])  # nodule center
        # 将标注转化为在图片中像素位置
        center = (center - origin) / spacing
        nodesCenter.append(center)
    if ifprint:
        for i, center in enumerate(nodesCenter):
            print("should:{}\t{}".format(i, center))

    return np.array(nodesCenter)


###
def nodeCubeCut(coor, img_array):
    cubexhalf = 16
    cubeyhalf = 16
    cubezhalf = 5
    cubes = np.zeros((coor.shape[0], 1, 2 * cubezhalf, 2 * cubexhalf, 2 * cubeyhalf))
    for i, c in enumerate(coor):
        bcx = int(np.rint(c[0]))
        bcy = int(np.rint(c[1]))
        bcz = int(np.rint(c[2]))
        # 因为不对img处理，这里仅采用浅拷贝即可
        # 在没做spaing之前，z不好控制
        # 暂采用10×32×32的size
        img = img_array[bcz - cubezhalf:bcz + cubezhalf, bcx - cubexhalf:bcx + cubexhalf,
              bcy - cubeyhalf:bcy + cubeyhalf]
        plt.imshow(img)
        cubes[i][0] = img
    # np.save(luna_path + 'cubes/' + outfilename + '.npy', cubes)
    return cubes


#####################
#
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


# df_node = pd.read_csv(luna_path + "csv/train/annotations.csv")
df_node = pd.read_csv(luna_path + "csv/val/annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()

cubes = np.array([])
cubes.shape = (0, 1, 10, 32, 32)
for img_file in tqdm(file_list):
    # debug时候针对特定mhd数据处理，并且作图；
    # 注意因为有for循环，debug模式一定要在debug模式下开启
    # debugMode = False
    debugMode = True

    if debugMode:
        # img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00192.mhd'
        # img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00168.mhd'
        img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00847.mhd'
        img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00092.mhd'
    # print("")
    # print("on mhd -- " + img_file)
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
        nodesCenter = extractNodeCenter(mini_df)
        # !正样本选取：
        newcube = nodeCubeCut(coor=nodesCenter, img_array=img_array)
        cubes = np.r_[cubes, newcube]

np.save(luna_path + 'cubes/' + 'posAll.npy', cubes)
