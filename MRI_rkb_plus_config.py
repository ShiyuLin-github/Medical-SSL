import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse


class rkbp_config:

    attr = 'class'
    # gpu_ids = [0, 1]
    gpu_ids = [0] # 没有多块GPU，做个修改看能不能跑通
    benchmark = False # 用于设置cudnn.benchmark
    manualseed = 666 # 随机数种子
    model = 'Simple' # 用于决定是否是复杂的模型，在源码中BROL和PCRL是复杂模型
    # network = 'unet_3d_rkbp' # 用于base_trainer中init_model中的get_networks
    network = 'VIT_3d'
    init_weight_type = 'kaiming' 
    note = "RKBPlus_240_240_155_MRI" # 修改标记

    # data
    train_fold = [0, 1, 2, 3, 4] # rkb_plus_pretask.py中定义的get_luna_list中用到
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 4
    input_size = [120, 120, 77] # 自己设置的适合切MRI尺寸的大小 # [64, 64, 16]  # [128, 128, 64]
    org_data_size = [240, 240, 155-1] #[320, 320, 74]
    train_dataset = 'MRI' #用于init_dataloader的get_dataloder_3D
    eval_dataset = 'MRI'
    im_channel = 1 #用于init_model中的get_networks
    order_class_num = 100 #这里的num之所以是100是因为最后要设置分类头对应排序的种数，如果8的阶乘全部用上就太大了，所以挑了个最大数
    k_permutations_path = r'datasets_3D\Rubik_cube\permutaions\permutations_hamming_max_100.npy'
    #k_permutation在Rubik_cube中的base_rkb_pretask中进行处理，里面存的是所有可能的排列组合顺序和对应标签顺序，比如[1,2,3,4...,6,7,8]为第0种顺序，1 到 8 的数字的排列方式有 40,320 种（8的阶乘）所以文件里存的应该是对应的最大100种，和最大1000种，可以调用出来看一下就知道了
    gaps = [6, 6, 4]
    num_grids_per_axis = 2

    # model pre-training
    train_batch = 4 #用于init_dataloader的get_dataloder_3D
    val_batch = 4
    optimizer = "adam"
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [250]#[200]
    num_workers = 1 #用于init_dataloader的get_dataloder_3D
    max_queue_size = num_workers * 4
    epochs = 1000
    save_model_freq = 50
    patience = 40 # 用于early stopping判断多少个epochs没有提升
    lr = 1e-3
    loss = 'ce' # 用于设定loss

    # logs
    resume = None #继续训练之前训练过的模型
    # pretrained_model = r'D:\Projects\Github_Local\checkpoints\MRI\VIT_3d_Simple_RKBPlus_240_240_155_MRI\20231212-224856\RCB_MRI.pth'  #用于trainer中的get_training_phase函数，如果不为none就开始finetuning

    pretrained_model = None

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = rkbp_config()
    Trainer = RKBPTrainer(config)
    Trainer.train()





