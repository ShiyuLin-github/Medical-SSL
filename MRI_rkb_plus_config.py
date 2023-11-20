import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse


class rkbp_config:

    attr = 'class'
    # gpu_ids = [0, 1]
    gpu_ids = [0] #没有多块GPU，做个修改看能不能跑通
    benchmark = False #用于设置cudnn.benchmark
    manualseed = 666 #随机数种子
    model = 'Simple' #用于决定是否是复杂的模型，在源码中BROL和PCRL是复杂模型
    network = 'unet_3d_rkbp' #用于init_model中的get_networks
    init_weight_type = 'kaiming'
    note = "RKBPlus_240_240_155_MRI" #修改标记

    # data
    train_fold = [0, 1, 2, 3, 4] #这部分利用fold没咋明白
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 4
    input_size = [240, 240, 155] # [64, 64, 16]  # [128, 128, 64]
    org_data_size = [320, 320, 74]
    train_dataset = 'MRI' #用于init_dataloader的get_dataloder_3D
    eval_dataset = 'MRI'
    im_channel = 1 #用于init_model中的get_networks
    order_class_num = 100
    k_permutations_path = "../datasets_3D/Rubik_cube/permutations/permutations_hamming_max_100.npy"
    gaps = [6, 6, 4]
    num_grids_per_axis = 2

    # model pre-training
    train_batch = 8 #用于init_dataloader的get_dataloder_3D
    val_batch = 8
    optimizer = "adam"
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [250]#[200]
    num_workers = 10 #用于init_dataloader的get_dataloder_3D
    max_queue_size = num_workers * 4
    epochs = 1000
    save_model_freq = 50
    patience = 40
    lr = 1e-3
    loss = 'ce' #用于设定loss

    # logs
    resume = None #继续训练之前训练过的模型
    pretrained_model = None #用于trainer中的get_training_phase函数

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





