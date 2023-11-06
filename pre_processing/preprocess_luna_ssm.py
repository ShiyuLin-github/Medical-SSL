# coding: utf-8

"""
Cropping the CT volumes in LUNA2016 for RPL/ROT/Jigsaw/RKB/RKB+ pretext tasks.
"""


import warnings

warnings.filterwarnings('ignore')
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from sklearn import metrics
from optparse import OptionParser
from glob import glob
from skimage.transform import resize
from utils.tools import save_np2nii

sys.setrecursionlimit(40000)


def generator_from_one_volume(img_array, org_data_size=[320, 320, 74]): #generator_from_one_volume 函数接受一个输入的 CT 体积（3D 图像数据）并进行处理。它确保 CT 体积中的百斯单元（HU）值在指定范围内（-1000 到 1000）。然后，它通过选择输入体积的中心区域来将体积裁剪到目标大小（320x320x74）。处理后的图像数据被标准化为介于 0 和 1 之间的范围，函数返回处理后的图像数据。
    size_x, size_y, size_z = img_array.shape

    hu_min = -1000.
    hu_max = 1000.
    img_array[img_array < hu_min] = hu_min
    img_array[img_array > hu_max] = hu_max
    img_array = 1.0 * (img_array - hu_min) / (hu_max - hu_min) #数据标准化，max跟min撞了怎么办？

    h1 = int(round((size_x - org_data_size[0]) / 2.))
    w1 = int(round((size_y - org_data_size[1]) / 2.))
    d1 = int(round((size_z - org_data_size[2]) / 2.))

    img_array = img_array[h1:h1 + org_data_size[0], w1:w1 + org_data_size[1], d1:d1 + org_data_size[2]]

    print(img_array.shape)

    return img_array


def get_self_learning_data(fold, data_path): #get_self_learning_data 函数：此函数处理 LUNA2016 数据集中特定折叠的 CT 体积。它接受两个参数：fold，它是要处理的折叠索引的列表，data_path，它是 LUNA2016 数据集的路径。对于指定折叠中的每个 CT 体积，它从“.mhd”文件中读取图像数据，对轴进行转置以匹配所需的方向（z、y、x），然后应用 generator_from_one_volume 函数来裁剪和标准化数据。处理后的数据以 NumPy 数组的形式保存在一个新目录中，每个体积的名称与原始文件相同，但使用“.npy”扩展名保存。输出显示处理的数据，包括裁剪体积的形状和体积中的最小值和最大值。

    for index_subset in fold:
        luna_subset_path = os.path.join(data_path, "subset" + str(index_subset))
        file_list = glob(os.path.join(luna_subset_path, "*.mhd"))
        print('{} files in fold {}'.format(len(file_list), index_subset))
        for img_file in tqdm(file_list):
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img)
            img_array = img_array.transpose(2, 1, 0)
            x = generator_from_one_volume(img_array, org_data_size=[320, 320, 74])
            # print(os.path.split(img_file)[1][:-4])
            save_path = '../../Data/LUNA2016_cropped_x320y320z74/' + "subset" + str(index_subset)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path+ '/' + str(os.path.split(img_file)[1][:-4]) + '.npy', x)
            print("cube: {} | {:.2f} ~ {:.2f}".format(x.shape, np.min(x), np.max(x)))


for fold in [0, 1, 2, 3, 4, 5, 6]:
    print(">> Fold {}".format(fold))
    get_self_learning_data([fold], data_path='../../Data/LUNA2016')

