# 肺癌检测
 100G肺部图像数据用于肺癌检测,训练多种model并ensemble

- 排名:**56**/2886

## U-NET

- 预处理阶段采用k-means进行2D图像分割并进行resize.

- 网络采用2D-unet训练. 

- 针对高虚警问题采用随机森林训练2级分类器,对unet候选结果再分类. 

- 结果: 该模型针对大结节效果好, 为利用3D信息以及对小结节有较好检出,自己实现一种3D-RCNN.

  ​

> 原unet做改动:
>
> 1. 网络有调整
> 2. roi有调整；修正resize拉伸bug
> 3. imagepreview用于查看unet输入图像情况
> 4. noROI用于不做lung mask，代替segment..ROI.py文件功能



## 3D-RCNN

- 预处理将不同病人肺片重采样到1mm. 
- 根据先验规则进行3Dregion proposal, 获得结节候选区为32*32*32的立方体, 正负例样本比例1:20. 
- 样本均衡.
- 基于keras训练3D-CNN.
- 结果: 该模型很好的弥补了unet小结节漏检问题

  ​



