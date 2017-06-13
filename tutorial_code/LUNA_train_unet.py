from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import optimizers
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K


# 设置工作路径
working_path = r"/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/minidata/tutorial/"
# working_path = r"/media/soffo/MEDIA/tcdata/val/"
# working_path = "/media/soffo/本地磁盘/tc/val/tutorial/"
working_path = "/media/soffo/本地磁盘/tc/train/tutorial/part1/"

# 待读取文件前缀（配合ROI操作之后）
# pre = 'noROI'
# pre = ''
pre = 'xf'
# pre = 'cj'
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # beta = 2.0
    # beta = 1
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # 模仿precision-recall方式修改loss
    return ((1.0 + beta ** 2) * intersection) / ((beta ** 2) * (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

# 貌似没什么卵用
def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    y_same = y_true_f * y_pred_f
    intersection = np.sum(y_same)
    # return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    # return 1 - dice_coef_np(y_true, y_pred)
    # 更正loss为0-1之间数值
    return 1 - dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    # 学习率小一点保险
    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy'])
    return model

# 大些的网络
def get_unet_():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy'])
    return model

# use_existing参数为是否利用已有权值训练网络
def train(use_existing=False):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train = np.load(working_path + pre + "trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path + pre + "trainMasks.npy").astype(np.float32)

    # 注意这里归一化问题：单张图还是数据集的归一化，待考证
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std
    # 曾经直接在train数据上验证
    # imgs_test = imgs_train
    # imgs_mask_test_true = imgs_mask_train
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    # model = get_unet_()           # 大网络训练
    model = get_unet()              # 小网络训练
    if use_existing:
        model.load_weights('./unet.hdf5')
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training. 
    #
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, \
                             write_images=True, embeddings_freq=1)
    model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=20, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])

    # 其实return的没啥卵用
    return model


def predict():
    # 可以只做预测
    model = get_unet()
    model.load_weights('./unet.hdf5')
    imgs_test = np.load(working_path + pre + "trainImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path + pre + "trainMasks.npy").astype(np.float32)
    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test, 1, 512, 512], dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i + 1]], verbose=1)[0]
        # print(imgs_mask_test[i][0])
        plt.subplots()
        plt.subplot(121)
        plt.title('original image')
        plt.imshow(imgs_test[i][0])

        plt.subplot(243)
        plt.title('labeled node mask')
        plt.imshow(imgs_mask_test_true[i][0])

        plt.subplot(244)
        plt.imshow(imgs_test[i][0]*imgs_mask_test_true[i][0])

        plt.subplot(247)
        plt.title('predicted node mask')
        plt.imshow(imgs_mask_test[i][0])

        plt.subplot(248)
        plt.imshow(imgs_test[i][0]*imgs_mask_test[i][0])
        # plt.colorbar()
        plt.show()

    np.save('masksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean += dice_coef_np(imgs_mask_test_true[i, 0], imgs_mask_test[i, 0])
    mean /= num_test
    print("Mean Dice Coeff : ", mean)


if __name__ == '__main__':
    # train(use_existing=False)
    # train(use_existing=True)
    predict()
