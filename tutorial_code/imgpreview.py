# encoding: utf-8

'''

@author: Soofo

@license: (C) Copyright 2017-2018, IECAS Limited.

@contact: iecaser@163.com

@software: pycharm

@file: imgpreview.py

@time: 17-6-1 下午4:13

@desc: 预览unet输入图片

'''

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

mode = input('''
    Chose mode:
        1. view unet input
        2. view image/lungmask before ROI
''')

if mode == '1':

    working_path = input("路径:")
    if not working_path.endswith('/'):
        working_path += '/'
    # working_path = '/media/soffo/本地磁盘/tc/val/tutorial/'
    # unet输入shape为798*1*512*512


    # 查看pre + trainImages.npy文件名的图片，pre为文件名前缀
    pre = 'noROI'
    # pre = 'xf'

    imgs_train = np.load(working_path + pre + "trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path + pre + "trainMasks.npy").astype(np.float32)
    for img, img_mask in zip(imgs_train, imgs_mask_train):
        plt.title('original_image')
        # img.shape为1*512*512,下面mask也是
        plt.imshow(img[0])
        plt.colorbar()

        plt.subplots()
        plt.subplot(131)
        plt.title('original_image')
        # img.shape为1*512*512,下面mask也是
        plt.imshow(img[0])
        plt.colorbar()
        # print(img[0])

        plt.subplot(132)
        plt.title('node_mask')
        plt.imshow(img_mask[0])
        plt.colorbar()
        # print(img_mask[0])

        plt.subplot(133)
        plt.title('original_image  ×  node_mask')
        plt.imshow(img_mask[0] * img[0])
        plt.colorbar()

        # plt.subplots()
        # plt.subplot(121)
        # plt.title('original_image hist')
        # plt.xlabel('max:{} mean:{} min:{}'.format(np.max(img[0]), np.round(np.mean(img[0])), np.min(img[0])))
        # plt.hist(img[0].flatten())
        #
        # plt.subplot(122)
        # plt.title('node_image hist')
        # plt.xlabel('max:{} mean:{} min:{}'.format(np.max(img[0]), np.round(np.mean(img[0])), np.min(img[0])))
        # plt.hist(img_mask[0].flatten())

        plt.show()
elif mode == '2':
    working_path = '/media/soffo/本地磁盘/tc/val/tutorial/'
    while True:
        img_file = input(working_path+'{文件}：')
        img_file = working_path+img_file
        # imgs = np.load(img_file).astype(np.float32)
        imgs = np.load(img_file)
        plt.subplot(131)
        plt.imshow(imgs[0])
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(imgs[1])
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(imgs[2])
        plt.colorbar()
        plt.show()
else:
    pass
pass
