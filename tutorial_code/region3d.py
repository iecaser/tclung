# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: region3d.py

@time: 17-6-22 下午5:42

@desc:

'''

from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from tutorial_code.lung_segmentation import *
from sklearn.cluster import DBSCAN

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
from tqdm import tqdm  # 进度条
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d(image, threshold=0):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    # verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -420, dtype=np.int8) + 1
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
    return binary_image
    plt.subplots()
    plt.imshow(binary_image[0])
    plt.show()
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    # 这段存在问题，会导致只剩一边肺部
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


# Some helper functions

def make_mask(center, diam, z, width, height, spacing, origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 5)
    v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    x_data = [x * spacing[0] + origin[0] for x in range(width)]
    y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
    return (mask)


def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min = np.min(matrix)
    m_max = np.max(matrix)
    matrix = matrix - m_min
    return (np.array(np.rint((matrix - m_min) / float(m_max - m_min) * 65535.0), dtype=np.uint16))


# ------------------------------------------------------
def ballProposal(segmented_lungs, nodesCenter):
    # -------- 下面都是提取亮斑的步骤 --------------
    segmented_lungs_content = 1 - segmented_lungs
    # 分离背景和结节必要步骤
    segmented_lungs_content = binary_erosion(segmented_lungs_content).astype(segmented_lungs_content.dtype)
    # segmented_lungs_content = binary_erosion(segmented_lungs_content).astype(segmented_lungs_content.dtype)

    # for cm in segmented_lungs_content:
    #     plt.subplots()
    #     plt.imshow(cm)
    # 结合图片考察亮斑（结节）查出情况
    for j in range(10):
        plt.subplots()
        plt.subplot(121)
        plt.imshow(segmented_lungs_content[150 - j])
        plt.subplot(122)
        plt.imshow(img_array[150 - j])
    # plt.show()

    contentLabels = measure.label(segmented_lungs_content)
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
    cr = []
    # 对照真结节，统计查出数目
    ballCount = 0
    for region in region3d:
        # 全部打印用于对照查找问题
        # print("{} - aera:{}\t - \tcentroid:{}\t - \tequivalent_d:{}\t - \textent:{}\t - bbox:{}".format(
        #     region.label, region.area, region.centroid, region.equivalent_diameter, region.extent,
        #     region.bbox))
        if 8 <= region.area <= 1000:
            if 1.5 <= region.equivalent_diameter <= 9.5:
                if 0.08 <= region.extent <= 0.95:
                    rb = region.bbox
                    a = rb[3] - rb[0]
                    b = rb[4] - rb[1]
                    c = rb[5] - rb[2]
                    rectCoef = a / b + b / c + c / a
                    if 3.0 <= rectCoef <= 8.0:
                        # print(rectCoef,end='  -  ')
                        xyz = np.array([region.centroid[2], region.centroid[1], region.centroid[0]])
                        # pxyz有几个真实结点，就有几个值
                        l2Distance = np.sum((nodesCenter - xyz) ** 2, axis=1)
                        isTrueBall = len(np.where(l2Distance < 10.0)[0]) > 0
                        if isTrueBall:  # 认为是一个点,即真结节是候选（愿景）
                            # 找出来的结节候选
                            ballCount += 1
                            # 打印找出的点，为了后续对比
                            print("found:\t\t{}".format(xyz))
                            # print(
                            #     "Found{}: \t\t\t\t\t{} - aera:{}\t - \tcentroid:{}\t - \tequivalent_d:{}\t - \textent:{}\t - bbox:{}".format(
                            #         ballCount, region.label, region.area, region.centroid, region.equivalent_diameter,
                            #         region.extent, region.bbox))
                            continue
                        cregion.append(region)
                        cxyz.append(xyz)
                        cr.append(region.equivalent_diameter)
                        # print("{} - aera:{}\t - \tcentroid:{}\t - \tequivalent_d:{}\t - \textent:{}\t - bbox:{}".format(
                        #     region.label, region.area, region.centroid, region.equivalent_diameter, region.extent,
                        #     region.bbox))

    cr = np.array(cr)
    cxyz = np.array(cxyz)
    print("---------  found/should-negative : {}/{}-{}  ---------".format(ballCount, len(nodesCenter),
                                                                          len(cregion) + ballCount))

    return ballCount, len(nodesCenter), len(cregion)


# ------------------------------------------------------

def extractNodeCenter(mini_df):
    # 每个node标记出位置，marker选用其他
    nodesCenter = []
    for node_idx, cur_row in mini_df.iterrows():
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
        # just keep 3 slices

        center = np.array([node_x, node_y, node_z])  # nodule center
        # 将标注转化为在图片中像素位置
        center = (center - origin) / spacing

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
    for i, center in enumerate(nodesCenter):
        print("should:{}\t{}".format(i, center))

    return np.array(nodesCenter)


############
#
# Getting list of image files
# luna_path = r"/media/soffo/MEDIA/tcdata/"
# luna_path = r"/media/soffo/本地磁盘/tc/val/"
# luna_path = r"/media/soffo/本地磁盘/tc/train/"
luna_path = r"/home/soffo/Documents/codes/minidata/"
luna_subset_path = luna_path + 'data/'
output_path = luna_path + 'tutorial/'
luna_subset_path = luna_path + 'data/'
output_path = luna_path + 'tutorial/'
file_list = glob(luna_subset_path + "*.mhd")


#####################
#
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


#
# The locations of the nodes
df_node = pd.read_csv(luna_path + "csv/train/annotations.csv")
# df_node = pd.read_csv(luna_path + "csv/val/annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()

#####
#
# Looping over the image files
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00003.mhd'
img_file = '/home/soffo/Documents/codes/minidata/data/LKDS-00001.mhd'
founds = 0
shoulds = 0
negatives = 0
for img_file in file_list:
    print("on img -- " + img_file)
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        # k = 144
        # img_array = img_array[kk:kk + 200]
        num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
        # go through all nodes (why just the biggest?)


        # img = img_array[0]
        # plt.imshow(img)
        # plt.colorbar()
        # # plot_3d(img_array, 400)
        segmented_lungs = segment_lung_mask(img_array, False)
        # plt.subplots()
        # plt.imshow(img * segmented_lungs[0])
        # plt.colorbar()
        # # plot_3d(segmented_lungs, 0)
        # # plt.show()



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
        # nodesCenter = np.array([[165, 162, 289], [151, 287, 293]])
        nodesCenter = extractNodeCenter(mini_df)
        found, should, negative = ballProposal(segmented_lungs=segmented_lungs, nodesCenter=nodesCenter)
        founds += found
        shoulds += should
        negatives += negative
        pass
print("")
print("------------------------  Total  --------------------------")
print("--------  founds/shoulds-negatives : {}/{}-{}  ---------".format(founds,shoulds,negatives))













        # center3d = region3d.
        # # contentMask 3d 准备完毕
        # # 下面进行xyzp提取，用作聚类4d数据集
        # cz, cx, cy = np.where(segmented_lungs_content > 0)
        # cp = img_array[cz, cx, cy]
        # # cdata shape为len×4
        # cdata = np.c_[cz, cx, cy, cp]
        # cdata = np.c_[cz, cx, cy]
        # # 参数待设置
        # db = DBSCAN(eps=7.5, min_samples=5)
        # db.fit(cdata)
        # labels = db.labels_
        # uniqueLables = set(labels)
        # # scatter 待续
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # colors = plt.cm.Spectral(np.linspace(0, 1, len(uniqueLables)))
        # for k, color in zip(uniqueLables, colors):
        #     alpha = 0.1
        #     marker = '.'
        #     if k == -1:
        #         # 不画
        #         continue
        #         # # 或者标黑
        #         color = 'k'
        #         alpha = 0.2
        #         marker = '.'
        #     tempData = cdata[labels == k]
        #     ax.scatter3D(tempData['x'], tempData['y'], tempData['z'], color=color, marker=marker, alpha=alpha)

        # plot_3d(segmented_lungs_fill, 0)
        # plot_3d(segmented_lungs_fill - segmented_lungs, 0)

        # github 9th
        # 下面留的两种分割方法较好
        # plt.subplots()
        # plt.imshow(img_array[0])
        # mask = segment_HU_scan_elias(img_array)
        # plt.subplots()
        # plt.imshow(mask[0])
        # plt.subplots()
        # plt.imshow(img_array[0]*mask[0])
        #
        # mask = segment_HU_scan_frederic(img_array)
        # plt.subplots()
        # plt.imshow(mask[0])
        # plt.subplots()
        # plt.imshow(img_array[0]*mask[0])
        # plt.show()


        # 3d scatter
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # z, x, y = np.where(img_array > 500)
        # ax.scatter3D(x, y, z, marker='.', alpha=0.1)
