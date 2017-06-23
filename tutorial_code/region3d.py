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
    binary_image = np.array(image > -320, dtype=np.int8) + 1
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


############
#
# Getting list of image files
# luna_path = r"/media/soffo/MEDIA/tcdata/"
luna_path = r"/media/soffo/本地磁盘/tc/val/"
luna_path = r"/media/soffo/本地磁盘/tc/train/"
# luna_path = r"/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/minidata/"
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

img_file = '/media/soffo/本地磁盘/tc/train/data/LKDS-00003.mhd'

mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file

if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
    # load the data once
    itk_img = sitk.ReadImage(img_file)
    img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
    # img_array = img_array[144:200]
    num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
    # go through all nodes (why just the biggest?)

    # k = 288
    k = 293
    img = img_array[k]
    plt.imshow(img)
    plt.colorbar()
    # plot_3d(img_array, 400)
    segmented_lungs = segment_lung_mask(img_array, False)
    plt.subplots()
    plt.imshow(img * segmented_lungs[k])
    plt.colorbar()
    # plot_3d(segmented_lungs, 0)
    # plt.show()

    # fill是需要的，不过这个函数不好；故采用3d 腐蚀/膨胀
    segmented_lungs_dilation = binary_dilation(segmented_lungs, structure=np.ones((5, 5, 5))).astype(segmented_lungs.dtype)
    plt.subplots()
    plt.imshow(segmented_lungs_dilation[k])
    segmented_lungs_dilation = binary_dilation(segmented_lungs_dilation, structure=np.ones((3, 3, 3))).astype(segmented_lungs.dtype)
    plt.subplots()
    plt.imshow(segmented_lungs_dilation[k])
    segmented_lungs_dilation = binary_dilation(segmented_lungs_dilation).astype(segmented_lungs.dtype)
    plt.subplots()
    plt.imshow(segmented_lungs_dilation[k])
    # segmented_lungs_dilation = binary_dilation(segmented_lungs_dilation).astype(segmented_lungs.dtype)
    # plt.subplots()
    # plt.imshow(segmented_lungs_dilation[10])



    segmented_lungs_erosion = binary_erosion(segmented_lungs).astype(segmented_lungs.dtype)
    plt.subplots()
    plt.imshow(segmented_lungs_erosion[k])

    segmented_lungs_erosion = binary_erosion(segmented_lungs_erosion).astype(segmented_lungs.dtype)
    plt.subplots()
    plt.imshow(segmented_lungs_erosion[k])

    # segmented_lungs_fill = segment_lung_mask(img_array, True)
    # plt.subplots()
    # plt.imshow(segmented_lungs_fill[10])
    # plt.colorbar()
    # plot_3d(segmented_lungs, 0)

    # 通过左查保留有用信息（尽量只包含血管/结节）
    content = segmented_lungs_dilation - segmented_lungs_erosion
    plt.subplots()
    plt.imshow(content[k])
    plt.colorbar()
    plt.subplots()

    # 未完待续，把第二大类的label指定为背景（也就是肺壁以及相邻）
    # ？？？？
    # 话说这里labels也算是聚类后的结果，考虑直接处理。不用kmean/dbscan等
    # 上述想法失败；考虑搜索3d检测球体代码
    labels = measure.label(content)
    l = labels[k]
    l[l>5]=5
    plt.subplots()
    plt.imshow(l)
    plt.colorbar()
    plt.title(len(list(l)))

    # content[labels==]=labels[0,0,0]
    img[content[k]==0]=-1000
    plt.imshow(img)
    plt.colorbar()
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

    # 每个node标记出位置，marker选用其他
    for node_idx, cur_row in mini_df.iterrows():
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
        # just keep 3 slices

        center = np.array([node_x, node_y, node_z])  # nodule center
        # 将标注转化为在图片中像素位置
        # rint 为就近取整
        v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
        vx = int(v_center[0])
        vy = int(v_center[1])
        vz = int(v_center[2])
        img = img_array[vz]
        mask = make_mask(center, diam, vz * spacing[2] + origin[2],
                         width, height, spacing, origin)
        plt.subplots()
        plt.subplot(131)
        plt.title('original_image')
        # img.shape为1*512*512,下面mask也是
        plt.imshow(img)
        plt.colorbar()
        # print(img[0])

        plt.subplot(132)
        plt.title('node_mask')
        plt.imshow(mask)
        plt.colorbar()
        # print(img_mask[0])

        plt.subplot(133)
        plt.title('original_image  ×  node_mask')
        plt.imshow(mask * img)
        plt.colorbar()

        # plt.show()

        # ax.scatter3D(x, y, z, color='r', marker='.')

    plt.show()
