import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms


class RKBBase(Dataset):
    def __init__(self, config, base_dir, flag='train'):
        self.config = config
        self.base_dir = base_dir
        self.all_images = []
        self.flag = flag
        self.crop_size = config.input_size
        self.org_data_size = config.org_data_size
        self.gaps = config.gaps
        self.num_grids_per_axis = config.num_grids_per_axis
        self.num_cubes = self.num_grids_per_axis ** 3

        self.order_num_classes = config.order_class_num
        self.rot_num_classes = self.num_cubes

        #不知道K_permutations干嘛的，但是没有path_file会报错，注释掉先吧
        # self.K_permutations = np.load(config.k_permutations_path)
        # assert self.order_num_classes == len(self.K_permutations)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        pass

    def crop_3d(self, image, flag, crop_size): # ...（对图像执行3D裁剪的方法）
        h, w, d = crop_size[0], crop_size[1], crop_size[2]
        h_old, w_old, d_old = image.shape[0], image.shape[1], image.shape[2]

        if flag == 'train': #如果 flag 是 'train'，则进行随机裁剪。在这种情况下，生成随机的裁剪起始点 (x, y, z)，h_old-h是为了确保裁剪框不超出图像边界。
            # crop random
            x = np.random.randint(0, 1 + h_old - h) #返回一个随机数或随机数数组(指定size时)
            y = np.random.randint(0, 1 + w_old - w)
            z = np.random.randint(0, 1 + d_old - d)
        else: #如果 flag 不是 'train'，则进行中心裁剪。计算裁剪起始点 (x, y, z) 使得裁剪的区域在图像中居中。
            # crop center
            x = int((h_old - h) / 2)
            y = int((w_old - w) / 2)
            z = int((d_old - d) / 2)

        return self.do_crop_3d(image, x, y, z, h, w, d)

    def do_crop_3d(self, image, x, y, z, h, w, d): #3D裁剪的辅助函数，确保整数和返回对应位置的3D块
        assert type(x) == int, x
        assert type(y) == int, y
        assert type(z) == int, z
        assert type(h) == int, h
        assert type(w) == int, w
        assert type(d) == int, d

        return image[x:x + h, y:y + w, z:z + d] #这边取的时最边界点加上crop_size的大小，解释了为何上述代码的定中心方式如此

    def crop_cubes_3d(self, image, flag, cubes_per_side, cube_jitter_xy=3, cube_jitter_z=3): #在3D图像中裁剪多个3D立方体的方法，jitter貌似是为了防止数据连续变化留出的间隔大小，从代码里可以看出如果有jitter则再次进行3D裁切，cubes_per_side应该是每个方向有多少个cube
        h, w, d = image.shape

        patch_overlap = -cube_jitter_xy if cube_jitter_xy < 0 else 0 #patch_overlap 计算了在裁剪时可能存在的重叠区域。如果 cube_jitter_xy 是负数，则 patch_overlap 采用该值；否则，为零。

        #h_grid、w_grid 和 d_grid 计算了每个立方体的网格大小，以确保裁剪时没有重叠区域。
        # 这里主要思想应该是把切出来的3D大块划分为各个小块,这里拿笔记做做数学题吧，好久没写了。
        h_grid = (h - patch_overlap) // cubes_per_side
        w_grid = (w - patch_overlap) // cubes_per_side
        d_grid = (d - patch_overlap) // cubes_per_side
        h_patch = h_grid - cube_jitter_xy
        w_patch = w_grid - cube_jitter_xy
        d_patch = d_grid - cube_jitter_z

        cubes = []
        for i in range(cubes_per_side):
            for j in range(cubes_per_side):
                for k in range(cubes_per_side):

                    p = self.do_crop_3d(image, #当i=0时，从0截取到第一个grid
                                   i * h_grid,
                                   j * w_grid,
                                   k * d_grid,
                                   h_grid + patch_overlap,
                                   w_grid + patch_overlap,
                                   d_grid + patch_overlap)

                    if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid: #如果有jitter，则对之前裁切好的3D块再次裁切以保留jitter大小的间隔
                        p = self.crop_3d(p, flag, [h_patch, w_patch, d_patch])

                    cubes.append(p)

        return cubes

    def rearrange(self, cubes, K_permutations):  # ...（根据排列重新排列立方体的方法）
        label = random.randint(0, len(K_permutations) - 1)
        # print('label', np.array(K_permutations[label]), label)
        return np.array(cubes)[np.array(K_permutations[label])], label

    def center_crop_xy(self, image, size): # 在image中间截一个size大小的缺口，画图很容易理解
        """CenterCrop a sample.
           Args:
              image: [D, H, W]
              label:[D, H, W]
              crop_size: the desired output size in the x-y plane
            Returns:
              out_image:[D, h, w]
              out_label:[D, h, w]
        """
        h, w, d = image.shape

        h1 = int(round((h - size[0]) / 2.)) #round() 函数将结果四舍五入为最接近的整数，确保裁剪区域的起始点是整数。
        w1 = int(round((w - size[1]) / 2.))

        image = image[h1:h1 + size[0], w1:w1 + size[1], :]
        return image

    def rotate(self, cubes): # ...（旋转3D立方体的方法）

        # multi-hot labels
        # [8, H, W, D]
        rot_cubes = copy.deepcopy(cubes) # 创建输入立方体的深层副本，以免修改原始数据
        hor_vector = [] # 记录水平旋转的向量
        ver_vector = [] # 记录垂直旋转的向量

        for i in range(self.num_cubes):
            p = random.random()  # 生成一个0到1之间的随机数
            cube = rot_cubes[i] # 获取当前处理的立方体
            # [H, W, D]
            if p < 1/3: # 如果 p 小于 1/3，表示进行水平旋转。将水平旋转的标志添加到 hor_vector，并沿x轴翻转180度。
                hor_vector.append(1)
                ver_vector.append(0)
                # rotate 180 along x axis
                rot_cubes[i] = np.flip(cube, (1, 2)) # 沿x轴翻转180度
            elif p < 2/3: #如果 p 大于等于 1/3 且小于 2/3，表示进行垂直旋转。将垂直旋转的标志添加到 ver_vector，并沿z轴翻转180度。
                hor_vector.append(0)
                ver_vector.append(1)
                # rotate 180 along z axis
                rot_cubes[i] = np.flip(cube, (0, 1)) # 沿z轴翻转180度

            else: #如果 p 大于等于 2/3，不进行旋转，标志都设为0。
                hor_vector.append(0)
                ver_vector.append(0)

        return rot_cubes, hor_vector, ver_vector

    def mask(self, cubes): # ...（对3D立方体应用掩码的方法）
        mask_vector = []
        masked_cubes = copy.deepcopy(cubes) # 创建一个输入立方体的深层副本，以免修改原始数据
        for i in range(self.num_cubes):
            cube = masked_cubes[i] # 获取当前处理的立方体
            if random.random() < 0.5: # 如果随机数小于0.5，应用掩码
                # mask
                mask_vector.append(1) # 记录掩码的标志为1
                R = np.random.uniform(0, 1, cube.shape) # 生成与立方体相同形状的随机数矩阵
                R = (R > 0.5).astype(np.int32) # 将大于0.5的值设置为1，否则为0，形成二值掩码
                masked_cubes[i] = cube * R # 将立方体与二值掩码相乘，进行掩码操作
            else:
                mask_vector.append(0) # 如果随机数大于等于0.5，不应用掩码，标志为0

        return masked_cubes, mask_vector