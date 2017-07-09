# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: region3d.py

@time: 17-6-22 下午5:42

@desc: 用于3d region proposal，待修正地方还有很多

'''

from __future__ import print_function, division
from scipy.ndimage.morphology import binary_erosion, binary_dilation
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
path = 'val/'
# path = 'train/'
path = 'test/'

luna_path = r"/media/soffo/本地磁盘/tc/" + path

luna_subset_path = luna_path + 'data/'
cubesize = np.load('/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/feng/cube.npy')
cubexhalf = cubesize[0]
cubeyhalf = cubesize[1]
cubezhalf = cubesize[2]
cubexhalf = 16
cubeyhalf = 16
cubezhalf = 16
file_list = glob(luna_subset_path + "*.mhd")


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, threshold=-600, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > threshold, dtype=np.int8) + 1
    # binary_image = np.array(image > 0.3*np.mean(image), dtype=np.int8) + 1
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1
    # 觉得后面没啥用，提前返回
    return 1 - binary_image


def extractNodeCenter(mini_df, origin, ifprint=False):
    # 每个node标记出位置，marker选用其他
    nodesCenter = []
    diams = []
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
        diams.append(diam)
    if ifprint:
        for i, center in enumerate(nodesCenter):
            print("should:{}\t{}".format(i, center))

    return np.array(nodesCenter), np.array(diams)


# ------------------------------------------------------
def ballProposal(segmented_lungs_content, nodesCenter, param, ifplot=False):
    # -------- 下面都是提取亮斑的步骤 --------------
    # 分离背景和结节必要步骤
    # erosion有助于使大结节“封口”，和背景区分开来
    for i in range(param.erosionTimes):
        segmented_lungs_content = binary_erosion(segmented_lungs_content).astype(segmented_lungs_content.dtype)

    # 结合图片考察亮斑（结节）查出情况
    # nthNode为待查看结节序号（结合log信息中should:nthNode）
    # nthNode = 2
    # nthNode = 1
    nthNode = 0
    # nthNode = 4
    # nthNode = 9
    # nthNode = 10


    # if ifplot:
    #     for j in range(20):
    #         plt.subplots()
    #         plt.subplot(121)
    #         plt.imshow(segmented_lungs_content[int(nodesCenter[nthNode][2]) + 10 - j])
    #         plt.subplot(122)
    #         plt.imshow(img_array[int(nodesCenter[nthNode][2]) + 10 - j])
    #         # plt.show()

    # contentLabels = measure.label(segmented_lungs_content, connectivity=1)
    # contentLabels = measure.label(segmented_lungs_content, connectivity=2)
    contentLabels = measure.label(segmented_lungs_content, connectivity=3)
    region3d = measure.regionprops(contentLabels)

    # check region 3d
    # for region in region3d:
    #     print("{} - aera:{} - centroid:{} - equivalent_d:{} - extent:{} - bbox:{}".format(region.label, region.area,
    #                                                                                       region.centroid,
    #                                                                                       region.equivalent_diameter,
    #                                                                                       region.extent, region.bbox))

    # 下面是当3d不好使的时候用的，每一帧做props，然后多帧通过聚类合并。
    # for i,cm in enumerate(segmented_lungs_content):
    #     cl = measure.label(cm)
    #     regions = measure.regionprops(cl)
    #     plt.subplot(121)
    #     plt.imshow(cm)
    #     plt.subplot(122)
    #     # 没搞清楚为何165,163,289处的结节被划分到了大背景，每次都这样？？？？！！！！
    #     plt.imshow(img_array[i]*cm)
    #     # for region in regions:
    #     #     print("{} - aera:{} - centroid:{} - equivalent_d:{} - extent:{} - bbox:{}".format(region.label,region.area,region.centroid,region.equivalent_diameter,region.extent,region.bbox))


    # 3d props搞定了，但是候选太多，需要进一步筛选
    # 规则自定
    # 1. 大小在一定范围
    #   - 等效半径
    #   - area
    # 2. 占空比在一定范围
    #   - 太大说明接近立方体而非球体（这个弱条件一般不予限定）
    #   - 太小说明形状奇特。
    # 3. bbox限定
    #   - 注意条状region的extent也可以较大，有必要直接根据bbox做长宽高比限定

    cregion = []
    cxyz = []
    dia = []
    # 对照真结节，统计查出数目
    ballCount = 0
    # if ifplot:
    #     print("{} - label:{}".format(nodesCenter[nthNode], contentLabels[int(np.rint(nodesCenter[nthNode][2]))][
    #         int(np.rint(nodesCenter[nthNode][1]))][int(np.rint(nodesCenter[nthNode][0]))]))
    for region in region3d:
        # 全部打印用于对照查找问题
        if ifplot:
            print("{} - aera:{}\t - \tcentroid:{}\t - \tequivalent_d:{}\t - \textent:{}\t - bbox:{}".format(
                region.label, region.area, region.centroid, region.equivalent_diameter, region.extent,
                region.bbox))
        # 条件1:坐标和边界
        if cubezhalf < region.centroid[0] < segmented_lungs_content.shape[0] - cubezhalf and cubeyhalf < \
                region.centroid[1] < segmented_lungs_content.shape[1] - cubeyhalf and cubexhalf < \
                region.centroid[2] < segmented_lungs_content.shape[2] - cubeyhalf:
            # 条件2:所占像素点区域大小
            if param.areamin <= region.area <= param.areamax:
                # 条件3:等效直径大小
                if param.dmin <= region.equivalent_diameter <= param.dmin + 35:
                    # 条件4:占空比
                    # 按理说经过多次erosion的，extent约束要适当加强
                    if param.extent <= region.extent:
                        rb = region.bbox
                        a = rb[3] - rb[0]
                        b = rb[4] - rb[1]
                        c = rb[5] - rb[2]
                        rectCoef = a / b + b / c + c / a
                        # 条件5:接近正立方体(均值不等式)
                        if 3.0 <= rectCoef <= 5.0:
                            # print(rectCoef,end='  -  ')
                            cregion.append(region)
                            xyz = np.array([region.centroid[2], region.centroid[1], region.centroid[0]])
                            cxyz.append(xyz)
                            dia.append(region.equivalent_diameter)
    dia = np.array(dia).reshape(len(dia), )
    cxyz = np.array(cxyz).reshape(len(cxyz), 3)
    imgshape = np.array(segmented_lungs_content.shape)
    rio = cxyz / imgshape[::-1]
    balls = Ball(coor=cxyz, dia=dia, rio=rio)

    # if ifplot:
    #     print("------------------ balls -------------------")
    #     for region in cregion:
    #         print("{} - aera:{}\t - \tcentroid:{}\t - \tequivalent_d:{}\t - \textent:{}\t - bbox:{}".format(
    #             region.label, region.area, region.centroid, region.equivalent_diameter, region.extent,
    #             region.bbox))

    return balls


# ballProposal基本参数
class Param:
    def __init__(self, erosionTimes=1, extent=0.1, areamin=5, areamax=15000, dmin=2.0):
        self.erosionTimes = erosionTimes
        self.extent = extent
        self.areamin = areamin
        self.areamax = areamax
        self.dmin = dmin


# 坐标和等效直径（分割有用的就这俩）
# 关于命名，真结节叫node，通过亮斑找出来的这一堆只能叫“球”
class Ball:
    def __init__(self, coor=np.array([]), dia=np.array([]), rio=np.array([])):
        self.coor = coor
        self.dia = dia
        self.rio = rio


# repetition check
def repCheck(ballOld, ballNew, protectDistance=10.0):
    # D = 15.0
    # D = 20.0
    if ballNew.coor.shape[0] > 0:
        ballNewLight = np.zeros(ballNew.coor.shape[0]).astype(bool)
        for i, xyzNew in enumerate(ballNew.coor):
            cnorm = np.linalg.norm(ballOld.coor - xyzNew, axis=1)
            isrep = np.sum(cnorm < protectDistance).astype(bool)
            ballNewLight[i] += isrep
        ballNew.coor = ballNew.coor[~ballNewLight]
        ballNew.dia = ballNew.dia[~ballNewLight]
        ballNew.rio = ballNew.rio[~ballNewLight]
        balls = Ball(coor=np.r_[ballOld.coor, ballNew.coor], dia=np.r_[ballOld.dia, ballNew.dia],
                     rio=np.r_[ballOld.rio, ballNew.rio])
        return balls
    else:
        return ballOld


# 其实基本逻辑同repCheck相似，不过return不同，有待抽取相同逻辑
# ball->node的check函数
def nodeCheck(ball, nodesCenter, protectDistance=10.0):
    nodesCenterLight = np.zeros(nodesCenter.shape[0]).astype(bool)
    nodeShould = len(nodesCenter)
    nodeFound = 0
    nodeNegative = 0
    if ball.coor.shape[0] > 0:
        # D-> protectDistance 相当于CFAR的保护单元，在此范围内都不作为负样本
        # 现采用方式为两坐标距离平方和<D认为是同一结节
        # 注！：后续可考虑修正为半径覆盖范围内认为同一结节
        # D = 15.0
        # D = 20.0
        coorLight = np.zeros(ball.coor.shape[0]).astype(bool)

        for i, nc in enumerate(nodesCenter):
            cnorm = np.linalg.norm(ball.coor - nc, axis=1)
            isFounds = (cnorm < protectDistance)
            isFound = np.sum(isFounds).astype(bool)
            coorLight += isFounds
            nodesCenterLight[i] += isFound
        nodeFound = np.sum(nodesCenterLight)
        # 找到的都扔掉
        ball.coor = ball.coor[~coorLight]
        ball.dia = ball.dia[~coorLight]
        nodeNegative = len(ball.coor)

    # 打印找出的点，为了后续对比
    print('''            \t{}
--------------------  found/should-negative : {}/{}-{}  --------------------
    '''.format(nodesCenterLight, nodeFound, nodeShould, nodeNegative))
    return nodeFound, nodeShould, nodeNegative


# ------------------------------------------------------




###
def cubeCut(coor, img_array, outfilename='test'):
    # coor = ball.coor
    cubesize = np.load('/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/feng/cube.npy')
    cubexhalf = cubesize[0]
    cubeyhalf = cubesize[1]
    cubezhalf = cubesize[2]
    cubexhalf = 16
    cubeyhalf = 16
    cubezhalf = 16
    cubes = np.zeros((coor.shape[0], 1, 2 * cubezhalf, 2 * cubexhalf, 2 * cubeyhalf))
    for i, c in enumerate(coor):
        bcx = int(np.rint(c[0]))
        bcy = int(np.rint(c[1]))
        bcz = int(np.rint(c[2]))
        # 因为不对img处理，这里仅采用浅拷贝即可
        # 在没做spaing之前，z不好控制
        # 暂采用10×32×32的size
        img = img_array[bcz - cubezhalf:bcz + cubezhalf, bcy - cubeyhalf:bcy + cubeyhalf,
              bcx - cubexhalf:bcx + cubexhalf]
        cubes[i][0] = img
    np.save(luna_path + 'cubes/' + outfilename + '.npy', cubes)
    # np.save(luna_path + 'dias/' + outfilename + '.npy', ball.dia)


def resample(imgs, spacing, order=2):
    # spacing 输入为xyz排序，变换为zyx
    spacing = np.array(list(reversed(spacing)))
    newShape = np.round(imgs.shape * spacing)
    resizeFactor = newShape / imgs.shape
    imgs = zoom(imgs, resizeFactor, mode='nearest', order=order)
    return imgs


#####################
#
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


founds = 0
shoulds = 0
negatives = 0
mode = 'test'
# mode = 'train'
oldfile = glob(luna_path + 'cubes/neg/*.npy')
ofnames = []
for ofile in oldfile:
    ofname = re.split('\.|/', ofile)[-2][3:]
    ofnames.append(ofname)
for img_file in tqdm(file_list):
    # debug时候针对特定mhd数据处理，并且作图；
    # 注意因为有for循环，debug模式一定要在debug模式下开启
    debugMode = False
    # debugMode = True

    if debugMode:
        # img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00192.mhd'
        # img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00168.mhd'
        # img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00847.mhd'
        img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00804.mhd'
    print("")
    print("on mhd -- " + img_file)
    mhdname = re.split('\.|/', img_file)[-2]
    if mhdname in ofnames:
        continue
    # load the data once
    itk_img = sitk.ReadImage(img_file)
    img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
    # k = 144
    # img_array = img_array[kk:kk + 200]
    num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
    img_array = resample(img_array, spacing=spacing, order=1)
    nodesCenter = []
    # for img in img_array:
    #     plt.subplots()
    #     plt.imshow(img)


    # 以下代码仅为初期探索，已移除
    # img_array = gaussian(img_array)
    # img_array = gaussian(img_array)
    # segmented_lungs = img_array > -600
    # plt.subplots()
    # plt.subplot(121)
    # plt.imshow(segmented_lungs[71])
    # plt.subplot(122)
    # plt.imshow(img_array[71])

    # plt.subplots()
    # plt.imshow(img * segmented_lungs[0])
    # plt.colorbar()
    # # plot_3d(segmented_lungs, 0)
    # # plt.show()
    #
    # # fill是需要的，不过这个函数不好；故采用3d 腐蚀/膨胀
    # segmented_lungs_dilation = binary_dilation(segmented_lungs, structure=np.ones((5, 5, 5))).astype(
    #     segmented_lungs.dtype)
    # plt.subplots()
    # plt.imshow(segmented_lungs_dilation[0])
    # segmented_lungs_dilation = binary_dilation(segmented_lungs_dilation, structure=np.ones((3, 3, 3))).astype(
    #     segmented_lungs.dtype)
    # plt.subplots()
    # plt.imshow(segmented_lungs_dilation[0])
    # segmented_lungs_dilation = binary_dilation(segmented_lungs_dilation).astype(segmented_lungs.dtype)
    # plt.subplots()
    # plt.imshow(segmented_lungs_dilation[0])
    # # segmented_lungs_dilation = binary_dilation(segmented_lungs_dilation).astype(segmented_lungs.dtype)
    # # plt.subplots()
    # # plt.imshow(segmented_lungs_dilation[10])
    #
    #
    # # 这里不需要了吗
    # # segmented_lungs_erosion = binary_erosion(segmented_lungs).astype(segmented_lungs.dtype)
    # segmented_lungs_erosion = segmented_lungs.copy()
    # # plt.subplots()
    # # plt.imshow(segmented_lungs_erosion[k])
    #
    # # segmented_lungs_erosion = binary_erosion(segmented_lungs_erosion).astype(segmented_lungs.dtype)
    # # plt.subplots()
    # # plt.imshow(segmented_lungs_erosion[k])
    #
    # # segmented_lungs_fill = segment_lung_mask(img_array, True)
    # # plt.subplots()
    # # plt.imshow(segmented_lungs_fill[10])
    # # plt.colorbar()
    # # plot_3d(segmented_lungs, 0)
    #
    #
    #
    # # 未完待续，把第二大类的label指定为背景（也就是肺壁以及相邻）
    # # ？？？？
    # # 话说这里labels也算是聚类后的结果，考虑直接处理。不用kmean/dbscan等
    # # 上述想法失败；考虑搜索3d检测球体代码
    # # content_labels = measure.label(content)
    # # erosion_labels = measure.label(segmented_lungs_erosion)
    #
    # # 试图框出有用信息范围3d
    # # regions = measure.regionprops(erosion_labels)
    #
    # # 通过左查保留有用信息（尽量只包含血管/结节）
    # # 这里的content是3d数据
    # segmented_lungs_content = segmented_lungs_dilation - segmented_lungs_erosion
    #
    # # for erosionMask, dilationMask, contentMask, img in zip(segmented_lungs_erosion, segmented_lungs_dilation,
    # #                                                        segmented_lungs_content, img_array):
    # for i, erosionMask in enumerate(segmented_lungs_erosion):
    #     erosionLabels = measure.label(erosionMask)
    #     regions = measure.regionprops(erosionLabels)
    #     if len(regions) == 0:
    #         continue
    #     min_row = 512
    #     max_row = 0
    #     min_col = 512
    #     max_col = 0
    #     for prop in regions:
    #         B = prop.bbox
    #         if min_row > B[0]:
    #             min_row = B[0]
    #         if min_col > B[1]:
    #             min_col = B[1]
    #         if max_row < B[2]:
    #             max_row = B[2]
    #         if max_col < B[3]:
    #             max_col = B[3]
    #     dilationMask = segmented_lungs_dilation[i]
    #     regionMask = np.zeros_like(dilationMask)
    #     regionMask[min_row:max_row, min_col:max_col] = 1
    #     # plt.subplots()
    #     # plt.subplot(121)
    #     # plt.imshow(erosionMask)
    #     # plt.subplot(122)
    #     # plt.imshow(regionMask)
    #     #
    #     # plt.subplots()
    #     # plt.imshow(dilationMask)
    #     # dilation 结果粗筛选
    #     dilationMask *= regionMask
    #     segmented_lungs_content[i] = dilationMask - erosionMask
    #     # plt.subplots()
    #     # plt.subplot(121)
    #     # plt.imshow(img)
    #     # plt.subplot(122)
    #     # plt.imshow(img*dilationMask)
    #     #
    #     # plt.subplots()
    #     # plt.subplot(121)
    #     # plt.imshow(contentMask)
    #     # plt.colorbar()
    #     # plt.subplot(122)
    #     # plt.imshow(contentMask*img)
    # # 为了尽量让结节和背景分离，因为measure的props方式，不能有一点点的直连，而dbscan耗时太久。
    # # 发现一个特点，先把孔扩张（对应mask的erosion），再在这一步缩小，相对于不这样做，前者更倾向于把孔变为球状。
    # # 破坏了实际形状，使mask更倾向于球状（是好是坏呢？毕竟只是mask，目标是检测出来所有可疑结点）

    # 将坐标提取归入extractNodeCenter函数



    # 参数设定
    # 试图用不erosion的进行小结节提取,多次erosion的进行大结节提取
    if path == 'train/':
        p0 = Param(erosionTimes=0, extent=0.2, areamin=8, areamax=1000, dmin=2.0)
        p1 = Param(erosionTimes=1, extent=0.3, areamin=5, areamax=1000, dmin=1.3)
        p2 = Param(erosionTimes=2, extent=0.4, areamin=4, areamax=1000, dmin=1.3)
        # p3 = Param(erosionTimes=3, extent=0.2, areamin=2, areamax=4000, dmin=1.3)
    # p5 = Param(erosionTimes=5, areamin=2, areamax=15000, dmin=1.3)
    # # 参数设定
    if path == 'test/' or path == 'val/':
        # 试图用不erosion的进行小结节提取,多次erosion的进行大结节提取
        p0 = Param(erosionTimes=0, extent=0.1, areamin=5, areamax=14000, dmin=1.3)
        p1 = Param(erosionTimes=1, extent=0.1, areamin=4, areamax=14000, dmin=1.3)
        p2 = Param(erosionTimes=2, extent=0.1, areamin=2, areamax=14000, dmin=1.3)
        p3 = Param(erosionTimes=3, extent=0.1, areamin=1, areamax=15000, dmin=1.3)
    # # p5 = Param(erosionTimes=5, areamin=2, areamax=15000, dmin=1.3)
    #     if path == 'train/' or path == 'val/':
    #     p0 = Param(erosionTimes=0, extent=0.3, areamin=10, areamax=3000, dmin=2.0)
    #     p1 = Param(erosionTimes=1, extent=0.3, areamin=5, areamax=3000, dmin=1.3)
    #     p2 = Param(erosionTimes=2, extent=0.4, areamin=4, areamax=3000, dmin=1.3)
    #     # p3 = Param(erosionTimes=3, extent=0.2, areamin=2, areamax=4000, dmin=1.3)
    # # p5 = Param(erosionTimes=5, areamin=2, areamax=15000, dmin=1.3)
    # # # 参数设定
    # if path == 'test/':
    #     # 试图用不erosion的进行小结节提取,多次erosion的进行大结节提取
    #     p0 = Param(erosionTimes=0, extent=0.1, areamin=5, areamax=14000, dmin=1.3)
    #     p1 = Param(erosionTimes=1, extent=0.1, areamin=4, areamax=14000, dmin=1.3)
    #     p2 = Param(erosionTimes=2, extent=0.1, areamin=2, areamax=14000, dmin=1.3)
    #     p3 = Param(erosionTimes=3, extent=0.1, areamin=1, areamax=15000, dmin=1.3)
    #     # p5 = Param(erosionTimes=5, areamin=2, areamax=15000, dmin=1.3)

    # 二值化门限设置
    th = -600

    # 这里的while循环为解决th门限对某些数据集太低，使得太多区域过了th

    # 主要耗时1
    segmented_lungs = segment_lung_mask(img_array, threshold=th, fill_lung_structures=False)
    # 主要耗时2
    ball0 = ballProposal(segmented_lungs_content=segmented_lungs, nodesCenter=nodesCenter, param=p0,
                         ifplot=debugMode)
    if ball0.coor.shape[0] < 2:
        print("-" * 40)
        print(img_file + '获取proposal失败!!')
        print("-" * 40)
        continue
    ball = Ball()
    # 实际上ballProposal中nodesCenter参数仅供debug模式作图查看真正结节前后的几帧图像，正式版将移除
    # 目前调用了3次ballProposal，实际上ball3作用很小，主要是为检出贴边较大结节，因为通过erosion或可将贴边的缺口闭合
    ball1 = ballProposal(segmented_lungs_content=segmented_lungs, nodesCenter=nodesCenter, param=p1,
                         ifplot=debugMode)
    ball2 = ballProposal(segmented_lungs_content=segmented_lungs, nodesCenter=nodesCenter, param=p2,
                         ifplot=debugMode)
    # ball3 = ballProposal(segmented_lungs_content=segmented_lungs, nodesCenter=nodesCenter, param=p3,
    #                      ifplot=debugMode)
    # ball5 = ballProposal(segmented_lungs_content=segmented_lungs, nodesCenter=nodesCenter, param=p5,
    #                      ifplot=debugMode)
    #
    # ball0采用p0参数，通常候选太多，当超过1000候选，可以做erosion大量减少；忽视ball0
    if ball0.coor.shape[0] > 1000:
        ball = ball1
    else:
        ball = repCheck(ball0, ball1)
    ball = repCheck(ball, ball2)
    # ball = repCheck(ball, ball3)
    # ball = repCheck(ball, ball5)
    # # 要将ball的coor and dia 存储起来
    # 保存世界位置coor和dia,用于回溯输出!
    np.save(luna_path + 'coor/coor{}.npy'.format(mhdname), ball.coor + origin)
    np.save(luna_path + 'dia/dia{}.npy'.format(mhdname), ball.dia)
    np.save(luna_path + 'rio/rio{}.npy'.format(mhdname), ball.rio)
    if path == 'val/' or path == 'test/':
        # 根据图片像素coor来cut
        cubeCut(coor=ball.coor, img_array=img_array, outfilename='neg/neg' + mhdname)
    else:
        # 下面是针对train和val的真假例区分
        df_node = pd.read_csv(luna_path + "csv/" + path + "annotations.csv")
        df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
        df_node = df_node.dropna()
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        if mini_df.shape[0] == 0:  # some files may not have a nodule--skipping those
            continue
        nodesCenter, dias = extractNodeCenter(mini_df, origin=origin, ifprint=True)
        # protectDistance 意义在于node 32像素范围内的ball都视为找到node
        # 这里于d无关，因为考虑到裁剪是按照固定大小32像素裁剪（前提是做spacing统一尺寸）
        nodeFound, nodeShould, nodeNegative = nodeCheck(ball=ball, nodesCenter=nodesCenter, protectDistance=32.0)
        # !负样本选取：
        # 上面对于每个mhd文件提取了ball坐标，下面进行cut
        # 认为nodeFound=0以及nodeNegative太大的，是因为数据不好
        # 应该对这些不好数据重新处理，这里为简便直接跳过；待后续修改。
        # if nodeFound > 0 and nodeNegative < 800:
        if nodeNegative < 800:
            # cubeCut函数里面有写文件操作
            cubeCut(coor=ball.coor, img_array=img_array, outfilename='neg/neg' + mhdname)
        else:
            print("-" * 40)
            print(img_file + 'proposal太大!!')
            print("-" * 40)
        # # 累加统计
        founds += nodeFound
        shoulds += nodeShould
        negatives += nodeNegative
        # 小结
        print("--------  founds/shoulds-negatives : {}/{}-{}  ---------".format(founds, shoulds, negatives))
# 总结
print("")
print("------------------------  Total  --------------------------")
print("---------   mhd counts: {}  ---------".format(len(file_list)))
print("--------  founds/shoulds-negatives : {}/{}-{}  ---------".format(founds, shoulds, negatives))
