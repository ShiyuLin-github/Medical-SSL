import copy
import random
import numpy as np
import torch
import torchio.transforms
from tqdm import tqdm
import os
import glob
import SimpleITK as sitk
from scipy import ndimage
from .base_rkb_pretask import RKBBase
import pandas as pd
import nibabel as nib


class RKBP_MRI_PretaskSet(RKBBase):

    def __init__(self, config, base_dir, flag, root_dir, csv_dir):
        super(RKBP_MRI_PretaskSet, self).__init__(config, base_dir, flag, root_dir, csv_dir)
        self.config = config
        self.flag = flag
        self.org_data_size = config.org_data_size
        ##添加MRI数据集位置
        self.root_dir = root_dir
        self.samples = os.listdir(root_dir)
        self.csv_file = pd.read_csv(csv_dir)

        if self.flag == "train": #这里修改的话不知道会不会有问题
            self.folds = config.train_fold
        else:
            self.folds = config.valid_fold

        # load data from .npy
        # self.get_luna_list()

        assert len(self.samples) != 0, "the images can`t be zero!"

    def __len__(self):
        # return len(self.all_images)
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = os.path.join(self.root_dir, self.samples[index])
        img_file = self.all_images[index]
        t1 = nib.load(os.path.join(sample_path, self.samples[index][:-6] + '_T1.nii.gz')).get_fdata()

        input_tensor = torch.Tensor(t1)
        # input: [240, 240, 155]
        # input = np.load(img_file)
        # input:  [320, 320, 74]

        if self.crop_size == [128, 128, 32]:
            # input: [276, 276, 74]
            input = self.center_crop_xy(input, [276, 276])

            # get all the num_grids **3 cubes
            all_cubes = self.crop_cubes_3d(
                input,
                flag=self.flag,
                cubes_per_side=self.num_grids_per_axis,
                cube_jitter_xy=10,
                cube_jitter_z=5,
            )
            # print(len(all_cubes), all_cubes[0].shape)

        elif self.crop_size == [64, 64, 16]:
            # input: [140, 140, 40]
            input = ndimage.zoom(input, [140 / 320, 140 / 320, 40 / 74], order=3)

            # get all the num_grids **3 cubes
            all_cubes = self.crop_cubes_3d(
                input,
                flag=self.flag,
                cubes_per_side=self.num_grids_per_axis,
                cube_jitter_xy=6,
                cube_jitter_z=4,
            )

        else:
            NotImplementedError("This crop size has not been configured yet")
            all_cubes = None

        # Task1: Permutate the order of cubes
        rearranged_cubes, order_label = self.rearrange(all_cubes, self.K_permutations)

        # Task2: Rotate each cube randomly.
        rotated_cubes, hor_label, ver_label = self.rotate(rearranged_cubes)

        # Task2: Mask each cube randomly.
        masked_cubes, mask_label = self.mask(rotated_cubes)

        final_cubes = np.expand_dims(np.array(masked_cubes), axis=1)

        return (
            torch.from_numpy(final_cubes.astype(np.float32)),
            torch.from_numpy(np.array(order_label)),
            torch.from_numpy(np.array(hor_label)).float(),
            torch.from_numpy(np.array(ver_label)).float(),
            torch.from_numpy(np.array(mask_label)).float(),
        )

    def get_luna_list(self):
        self.all_images = []
        for index_subset in self.folds:
            luna_subset_path = os.path.join(self.base_dir, "subset" + str(index_subset))
            file_list = glob.glob(os.path.join(luna_subset_path, "*.npy"))
            # save the paths
            for img_file in tqdm(file_list):
                self.all_images.append(img_file)
        # x_train: (445)
        # x_valid: (178)
        return
    