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
from datasets_3D.Classification.base_classification import ClassificationBase
import pandas as pd
import nibabel as nib


class MRI_Classification(ClassificationBase):

    def __init__(self, config, base_dir, csv_dir,flag):
        super(MRI_Classification, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        ##添加MRI数据集位置
        self.base_dir = base_dir
        self.samples = os.listdir(self.base_dir)
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
        
        # Load each MRI sequence from the nii files,-6是除去文件夹名称里的_nifti
        t1 = nib.load(os.path.join(sample_path, self.samples[index][:-6] + '_T1.nii.gz')).get_fdata()
        t1c = nib.load(os.path.join(sample_path, self.samples[index][:-6] + '_T1c.nii.gz')).get_fdata()
        t2 = nib.load(os.path.join(sample_path, self.samples[index][:-6] + '_T2.nii.gz')).get_fdata()
        flair = nib.load(os.path.join(sample_path, self.samples[index][:-6] + '_FLAIR.nii.gz')).get_fdata()
        asl = nib.load(os.path.join(sample_path, self.samples[index][:-6] + '_ASL.nii.gz')).get_fdata()

        # Combine the four MRI sequences into a single input tensor
        input_tensor = torch.Tensor(np.stack([t1, t1c, t2, flair, asl], axis=0))
        #input_tensor = input_tensor.permute(0,3,1,2) #将维度重排为 [C, D, H, W]，有了transform就不需要在这里重排了
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        #从csv文件中读取label
        # 原ID比csv文件中的ID多了个0，比如UCSF-PDGM-0004，csv中是UCSF-PDGM-004。所以要修改一下
        id = self.samples[index][:-6]
        id_fit = id[0:-4] + id[-3:]
        image_name = id_fit
        label = self.csv_file.loc[self.csv_file['ID'] == id_fit, 'WHO CNS Grade'].values[0]
        label = label - 2 #从[2,3,4]转为[0,1,2]
        #PyTorch会自动把整数型的label转为one-hot型，用于计算CE loss这里需要确保label是从0开始的,from深入浅出pytorch
        
        # Return the input tensor and any additional labels or targets
        return input_tensor, label, image_name  # label是你的样本的标签，需要自己定义

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
    