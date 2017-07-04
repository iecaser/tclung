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
from scipy.ndimage.interpolation import zoom

############
# 全局

# luna_path = r"/media/soffo/本地磁盘/tc/val/"
luna_path = r"/media/soffo/本地磁盘/tc/train/"

luna_subset_path = luna_path + 'data/'
output_path = luna_path + 'tutorial/'
file_list = glob(luna_subset_path + "*.mhd")
cubexhalf = 16
cubeyhalf = 16
cubezhalf = 16


###
def extractNodeCenter(mini_df, origin, ifprint=False):
    # 每个node标记出位置，marker选用其他
    nodesCenter = []
    for node_idx, cur_row in mini_df.iterrows():
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]

        center = np.array([node_x, node_y, node_z])  # nodule center
        # 将标注转化为在图片中像素位置
        # !注意center,origin,spacing都是xyz排序
        # center = (center - origin) / spacing
        # 注意当做了resample之后，这里需要做此更改
        center = (center - origin)
        # 下面为以前程序，这里暂时不需要
        # # rint 为就近取整
        # v_center = np.rint(center)  # nodule center in voxel space (still x,y,z ordering)
        # vx = int(v_center[0])
        # vy = int(v_center[1])
        # vz = int(v_center[2])
        # img = img_array[vz]
        # mask = make_mask(center, diam, vz * spacing[2] + origin[2],
        #                  width, height, spacing, origin)
        # plt.subplots()
        # plt.subplot(131)
        # plt.title('original_image')
        # # img.shape为1*512*512,下面mask也是
        # plt.imshow(img)
        # plt.colorbar()
        # # print(img[0])
        #
        # plt.subplot(132)
        # plt.title('node_mask')
        # plt.imshow(mask)
        # plt.colorbar()
        # # print(img_mask[0])
        #
        # plt.subplot(133)
        # plt.title('original_image  ×  node_mask')
        # plt.imshow(mask * img)
        # plt.colorbar()
        #
        # # plt.show()
        # # ax.scatter3D(x, y, z, color='r', marker='.')
        nodesCenter.append(center)
    if ifprint:
        for i, center in enumerate(nodesCenter):
            print("should:{}\t{}".format(i, center))

    return np.array(nodesCenter)


def nodeCubeCut(coor, img_array, expandCoef=8, vibration=5):
    # -----------------------------------
    # 1.注意到共有(2*vibration+1)**3中排列方式
    #   expandCoef应尽量大于上述数值
    # 2.expandCoef取值为>=1整数，为扩充数据倍数，如果=1，则不抖动
    # 3. vibration为抖动范围。注意要和裁剪的边界处理结合起来，防止裁剪后不是立方体bug
    # -----------------------------------

    i = 0
    cubelen = int(expandCoef * coor.shape[0])
    cubes = np.zeros((cubelen, 1, 2 * cubezhalf, 2 * cubexhalf, 2 * cubeyhalf))
    for c in coor:
        bcx = int(np.rint(c[0]))
        bcy = int(np.rint(c[1]))
        bcz = int(np.rint(c[2]))
        # +-vibration的抖动范围
        bcv = np.random.randint(-vibration, vibration + 1, (expandCoef, 3))
        # 要有不移动的裁剪
        bcv[0][:] = 0
        # 因为不对img处理，这里仅采用浅拷贝即可
        # 在没做spaing之前，z不好控制
        # 暂采用10×32×32的size
        for j in range(expandCoef):
            z1 = bcz - cubezhalf + bcv[j][2]
            z2 = bcz + cubezhalf + bcv[j][2]
            y1 = bcy - cubeyhalf + bcv[j][1]
            y2 = bcy + cubeyhalf + bcv[j][1]
            x1 = bcx - cubexhalf + bcv[j][0]
            x2 = bcx + cubexhalf + bcv[j][0]
            if z1 < 0:
                z1 = 0
                z2 = cubezhalf * 2
            if z2 > img_array.shape[0]:
                z2 = img_array.shape[0]
                z1 = z2 - cubezhalf * 2
            if y1 < 0:
                y1 = 0
                y2 = cubeyhalf * 2
            if y2 > img_array.shape[1]:
                y2 = img_array.shape[1]
                y1 = y2 - cubeyhalf * 2
            if x1 < 0:
                x1 = 0
                x2 = cubexhalf * 2
            if x2 > img_array.shape[2]:
                x2 = img_array.shape[2]
                x1 = x2 - cubexhalf * 2

            img = img_array[z1:z2, y1:y2, x1:x2]
            if np.sum(img.shape)<32*3:
                print(i)
                print(j)
                print(bcx)
                print(bcy)
                print(bcz)
                print(bcv)
            else:
                cubes[i][0] = img
            i += 1
    return cubes


#####################
#
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


def resample(imgs, spacing, order=2):
    # spacing 输入为xyz排序，变换为zyx
    spacing = np.array(list(reversed(spacing)))
    newShape = np.round(imgs.shape * spacing)
    resizeFactor = newShape / imgs.shape
    imgs = zoom(imgs, resizeFactor, mode='nearest', order=order)
    return imgs


def poscubeCut():
    df_node = pd.read_csv(luna_path + "csv/train/annotations.csv")
    # df_node = pd.read_csv(luna_path + "csv/val/annotations.csv")
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()
    cubes = np.array([])
    cubes.shape = (0, 1, cubezhalf * 2, cubexhalf * 2, cubeyhalf * 2)
    for img_file in tqdm(file_list):
        # debug时候针对特定mhd数据处理，并且作图；
        # 注意因为有for循环，debug模式一定要在debug模式下开启
        debugMode = False
        # debugMode = True

        if debugMode:
            # img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00192.mhd'
            # img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00168.mhd'
            # img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00847.mhd'
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
            img_array = resample(img_array, spacing=spacing, order=1)
            nodesCenter = extractNodeCenter(mini_df, origin=origin)
            # !正样本选取：
            newcube = nodeCubeCut(coor=nodesCenter, img_array=img_array, expandCoef=8, vibration=5)
            cubes = np.r_[cubes, newcube]
    np.save(luna_path + 'cubes/' + 'posAll.npy', cubes)
    return cubes


# run
cubes = poscubeCut()
# 因为cubesshape为m*1*z*x*y
#         [m,1,z,x,y]
# !暂未考虑做肺的上下颠倒(z轴未动)
# x轴对称
xcubes = cubes[:, :, :, ::-1]
np.save(luna_path + 'cubes/' + 'posxcubes.npy', cubes)
# y轴对称
ycubes = cubes[:, :, :, :, ::-1]
np.save(luna_path + 'cubes/' + 'posycubes.npy', cubes)
# xy都对称 <=> 旋转180°
xycubes = cubes[:, :, :, ::-1, ::-1]
np.save(luna_path + 'cubes/' + 'posxycubes.npy', cubes)
