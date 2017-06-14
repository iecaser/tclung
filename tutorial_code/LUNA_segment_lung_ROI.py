import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import matplotlib.pyplot as plt

# working_path = "/media/soffo/本地磁盘/tc/train/tutorial/part2/"
working_path = "/media/soffo/本地磁盘/tc/train/tutorial/part2/"
working_path = "/media/soffo/本地磁盘/tc/val/tutorial/"
# working_path = "/home/soffo/Documents/codes/DSB3Tutorial/tutorial_code/minidata/tutorial/"
# working_path = "/media/soffo/MEDIA/tcdata/tutorial/"
file_list = glob(working_path + "images_*.npy")
ifplot = False

for img_file in file_list:
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    # img_file = r'/media/soffo/本地磁盘/tc/train/tutorial/images_0191_0295.npy'
    imgs_to_process = np.load(img_file).astype(np.float64)
    print("on image", img_file)
    try:
        for i in range(len(imgs_to_process)):
            img = imgs_to_process[i]
            # Standardize the pixel values
            mean = np.mean(img)
            std = np.std(img)
            img = img - mean
            img = img / std
            # Find the average pixel value near the lungs
            # to renormalize washed out images
            middle = img[100:400, 100:400]
            mean = np.mean(middle)
            max = np.max(img)
            min = np.min(img)
            # To improve threshold finding, I'm moving the
            # underflow and overflow on the pixel spectrum
            img[img == max] = mean
            img[img == min] = mean
            #
            # Using Kmeans to separate foreground (radio-opaque tissue)
            # and background (radio transparent tissue ie lungs)
            # Doing this only on the center of the image to avoid
            # the non-tissue parts of the image as much as possible
            #
            kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
            centers = sorted(kmeans.cluster_centers_.flatten())
            threshold = np.mean(centers)
            thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
            #
            # I found an initial erosion helful for removing graininess from some of the regions
            # and then large dialation is used to make the lung region
            # engulf the vessels and incursions into the lung cavity by
            # radio opaque tissue
            #
            eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
            dilation = morphology.dilation(eroded, np.ones([10, 10]))
            #
            #  Label each region and obtain the region properties
            #  The background region is removed by removing regions
            #  with a bbox that is to large in either dimnsion
            #  Also, the lungs are generally far away from the top
            #  and bottom of the image, so any regions that are too
            #  close to the top and bottom are removed
            #  This does not produce a perfect segmentation of the lungs
            #  from the image, but it is surprisingly good considering its
            #  simplicity.
            #
            labels = measure.label(dilation)
            label_vals = np.unique(labels)
            regions = measure.regionprops(labels)
            good_labels = []
            for prop in regions:
                B = prop.bbox
                if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                    # 不同的连通区域产生了不同的label
                    # 当label代表的区域bbox足够，认为是有有用信息的区域
                    good_labels.append(prop.label)
            mask = np.ndarray([512, 512], dtype=np.int8)
            mask[:] = 0
            #
            #  The mask here is the mask for the lungs--not the nodes
            #  After just the lungs are left, we do another large dilation
            #  in order to fill in and out the lung mask
            #
            # 前面mask为0
            for N in good_labels:
                # 认为有用的区域是连通的，切足够大的
                # 将所有这种连通区域都包含进来，叠加为mask
                # 其余的边界/零散信息处mask仍为0
                mask = mask + np.where(labels == N, 1, 0)
            # mask又做了一次膨胀
            mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
            imgs_to_process[i] = mask
        np.save(img_file.replace("images", "lungmask"), imgs_to_process)
    except:
        print('----- {} 维度不匹配 -----'.format(img_file))

#
#    Here we're applying the masks and cropping and resizing the image
#


file_list = glob(working_path + "lungmask_*.npy")
out_images = []  # final set of images
out_nodemasks = []  # final set of nodemasks
for fname in file_list:
    print("working on file ", fname)
    imgs_to_process = np.load(fname.replace("lungmask", "images"))
    masks = np.load(fname)
    node_masks = np.load(fname.replace("lungmask", "masks"))
    for i in range(len(imgs_to_process)):
        mask = masks[i]
        node_mask = node_masks[i]
        img = imgs_to_process[i]
        # plt.imshow(img)
        # plt.colorbar()
        new_size = [512, 512]  # we're scaling back up to the original size of the image
        img = mask * img  # apply lung mask
        # 经测试不resize，用min和max都不如原图理想。其中min收敛到一般开始发散，loss很大，但是acc0.9
        # 以下为各种mask方式，最终没用mask，制作resize
        # img[mask == 0] = np.min(img)
        # img[mask == 0] = np.max(img)
        # noise = np.mean(img)*np.random.rand(img.shape[0],img.shape[1])
        # img[mask == 0] = noise[mask==0]

        # 作图观察
        # 注意要深度copy，为忽略下述对img的处理（会影响到imgxf）
        # imgxf = img.copy()
        # plt.subplots()
        # plt.imshow(imgxf)
        # plt.colorbar()
        # plt.subplots()
        # plt.hist(imgxf.flatten())
        # plt.show()
        #
        # renormalizing the masked image (in the mask region)
        #
        # new_mean = np.mean(img[mask > 0])
        # new_std = np.std(img[mask > 0])
        #
        #  Pulling the background color up to the lower end
        #  of the pixel range for the lungs
        #
        # old_min = np.min(img)  # background color
        # img[img == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
        # img = img - new_mean
        # img = img / new_std
        # make image bounding box  (min row, min col, max row, max col)
        # 利用mask连通性，选取集中感兴趣区域（ROI）
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        #
        # Finding the global min and max row over all regions
        #
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        width = max_col - min_col
        height = max_row - min_row
        # 下面确保图像是正方形图，以长边为准
        # 这里有个问题：下面的max_row或可大于512，这样子结果仍然不是方形，在下面的resize之后，会对图像拉伸；当然nodemask也会同时被拉伸，影响应该不是很大。
        if width > height:
            max_row = min_row + width
            # 为保证方形不拉伸
            overCut = max_row - 512
            if overCut > 0:
                max_row -= overCut
                min_row -= overCut
        else:
            max_col = min_col + height
            # 为保证方形不拉伸
            overCut = max_col - 512
            if overCut > 0:
                max_col -= overCut
                min_col -= overCut
        # 
        # cropping the image down to the bounding box for all regions
        # (there's probably an skimage command that can do this in one line)
        # 
        img = img[min_row:max_row, min_col:max_col]
        mask = mask[min_row:max_row, min_col:max_col]
        # imgxf = np.zeros((512,512))
        # maskxf = imgxf.copy()
        # maskxf[min_row:max_row, min_col:max_col] = node_mask[min_row:max_row, min_col:max_col]
        # imgxf[min_row:max_row, min_col:max_col]=img

        # 这里的try是因为lungmask_0111_0060.npy报错，经imagepreview查看
        # 该lung分割完全错误，没有debug具体引起错误原因，基于该样本错误原因，
        # 暂采取去除“分割”不好的样本原则。
        try:
            if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
                pass
            else:
                # moving range to -1 to 1 to accomodate the resize function
                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = img / (max - min)
                # 这一步将小图（只有感兴趣区域所以小）放大成512*512的图
                # ROI的拉伸
                new_img = resize(img, [512, 512])
                # nodeMask按照相同的裁剪方式（孔位置会对应缩放）
                new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], [512, 512])
                new_node_mask[new_node_mask != 0] = 1
                # 做图修改
                if ifplot:
                    plt.subplots()
                    plt.subplot(131)
                    # plt.imshow(imgxf)
                    plt.imshow(new_img)
                    plt.colorbar()
                    plt.subplot(132)
                    # plt.imshow(maskxf)
                    plt.imshow(new_node_mask)
                    plt.colorbar()
                    plt.subplot(133)
                    plt.hist(new_img.flatten())
                    plt.show()

                # 将ROI后的图片和nodeMask append 到list
                out_images.append(new_img)
                # out_images.append(imgxf)
                out_nodemasks.append(new_node_mask)
                # out_nodemasks.append(maskxf)
        except:
            pass
num_images = len(out_images)
#
#  Writing out images and masks as 1 channel arrays for input into network
#

# ROI后的图像（这里num_images按理说是3的倍数）
# numImages*1*512*512 的四维尺寸
final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
final_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
for i in range(num_images):
    # [i,0]这个0是因为多了一维,是为了卷积层的统一（后面将有多个卷积核）
    final_images[i, 0] = out_images[i]
    final_masks[i, 0] = out_nodemasks[i]

# rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
# 等价于下面式子。关键是replace参数指定不重复，即重排列
# rand_i = np.random.choice(num_images, num_images,replace=False)
# test_i = int(0.2*num_images)
# np.save(working_path+"trainImages.npy",final_images[rand_i[test_i:]])
# np.save(working_path+"trainMasks.npy",final_masks[rand_i[test_i:]])
# np.save(working_path+"testImages.npy",final_images[rand_i[:test_i]])
# np.save(working_path+"testMasks.npy",final_masks[rand_i[:test_i]])
# 自命名的xf前缀，用于区分不同方法ROI结果
np.save(working_path + "xftrainImages.npy", final_images)
np.save(working_path + "xftrainMasks.npy", final_masks)
# np.save(working_path + "testImages.npy", final_images)
# np.save(working_path + "testMasks.npy", final_masks)
