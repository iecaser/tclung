# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: noROI.py

@time: 17-6-5 下午5:32

@desc: 不做ROI，替换LUNA_segment_lung_ROI文件功能

'''
from glob import glob
import numpy as np

working_path = "/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/minidata/tutorial/"
working_path = "/media/soffo/本地磁盘/tc/val/tutorial/"
# working_path = "/media/soffo/MEDIA/tcdata/tutorial/"
file_list = glob(working_path + "images_*.npy")
out_images = []
out_nodemasks = []
for img_file in file_list:
    print("on image - ", img_file)
    imgs = np.load(img_file).astype(np.float64)
    node_masks = np.load(img_file.replace("images", "masks"))
    for i in range(len(imgs)):
        out_images.append(imgs[i])
        out_nodemasks.append(node_masks[i])

num_images = len(out_images)
final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
final_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
for i in range(num_images):
    # [i,0]这个0是因为多了一维,是为了卷积层的统一（后面将有多个卷积核）
    try:
        final_images[i, 0] = out_images[i]
        final_masks[i, 0] = out_nodemasks[i]
    except:
        pass
np.save(working_path + "noROItrainImages.npy", final_images)
np.save(working_path + "noROItrainMasks.npy", final_masks)
