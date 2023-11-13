from datasets_3D.paths import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets_3D.Seg import SegmentationLunaSet,\
    SegmentationLiTSTrainSet, SegmentationLiTSTestset, SegmentationMSDLIVERTrainSet, SegmentationMSDLIVERTestset
from datasets_3D.MG import MGLunaPretaskSet
from datasets_3D.PCRL import PCRLLunaPretaskSet_ORIG, PCRLLunaPretaskSet
from datasets_3D.Classification import ClassificationLUNASet
from datasets_3D.AE import AELunaPretaskSet
from datasets_3D.Rubik_cube import JigSawLunaPretaskSet,RKBLunaPretaskSet, RKBPLunaPretaskSet
from datasets_3D.CL import CLLunaPretaskSet
from datasets_3D.PTP import RotLunaPretaskSet, RPLLunaPretaskSet, RPLLunaPretaskSet_v2, RotLunaPretaskSet_v2

datasets_dict_3D = {
    'luna_cl_pretask': CLLunaPretaskSet,
    'luna_pcrl_pretask': PCRLLunaPretaskSet,
    'luna_pcrl_pretask_org': PCRLLunaPretaskSet_ORIG,
    'luna_mg_pretask': MGLunaPretaskSet,
    'luna_ae_pretask': AELunaPretaskSet,
    'luna_rot_pretask': RotLunaPretaskSet,
    'luna_rot_pretask_v2': RotLunaPretaskSet_v2,
    'luna_rpl_pretask': RPLLunaPretaskSet,
    'luna_rpl_pretask_v2': RPLLunaPretaskSet_v2,
    'luna_rkb_pretask': RKBLunaPretaskSet,
    'luna_rkbp_pretask': RKBPLunaPretaskSet,
    'luna_jigsaw_pretask': JigSawLunaPretaskSet,
    # "MRI": MRI, #新增我的data类

    'luna_ncs': SegmentationLunaSet,
    'luna_ncc': ClassificationLUNASet,
    'lits_seg_train':SegmentationLiTSTrainSet,
    'lits_seg_test':SegmentationLiTSTestset,
    'lits_seg_liver_tumor_train':SegmentationLiTSTrainSet,
    'lits_seg_liver_tumor_test': SegmentationLiTSTestset,
    'msd_liver_seg_train': SegmentationMSDLIVERTrainSet,
    'msd_liver_seg_test': SegmentationMSDLIVERTestset,
    'msd_liver_seg_train_down2': SegmentationMSDLIVERTrainSet,
    'msd_liver_seg_test_down2': SegmentationMSDLIVERTestset
}
#datasets_dict_3D 是一个字典，它将数据集的名称映射到相应的数据集类。这个字典包含了各种数据集名称和对应的数据集类


def get_dataloder_3D(args, flag="train", drop_last=True): #get_dataloder_3D 函数是一个用于获取数据加载器的函数。它接受参数 args，其中包括训练或测试标志 flag、是否丢弃最后一个批次的标志 drop_last。如果 flag 为 "train"，则表示获取训练数据加载器。根据 args.train_dataset 指定的数据集名称，从 datasets_dict_3D 中获取相应的数据集类，然后初始化数据集实例。接着设置批量大小（batch_size）、是否打乱数据（shuffle）、多线程加载数据的工作进程数（num_workers）以及是否将数据加载到 GPU 的固定内存中（pin_memory）。如果 flag 不是 "train"，则表示获取测试数据加载器，执行类似的步骤。最后，使用 torch.utils.data.DataLoader 创建数据加载器对象，该对象用于加载数据集，并返回数据集实例和数据加载器。
    '''
    :return: the dataloader of special datasets
    '''

    if flag == "train":
        print('******Building training dataloder******')
        datasets_name = args.train_dataset
        assert datasets_name in datasets_dict_3D.keys(), "The dataset use {} is not exist ".format(datasets_name)
        root = Path.db_root_dir(datasets_name) #在path.py里，从dataset名称到具体本地路径
        dataset = datasets_dict_3D[datasets_name](config=args, base_dir=root, flag=flag)
        batch_size = args.train_batch
        shuffle = True
        num_workers = args.num_workers
        pin_memory = True
    else:
        print('******Building test dataloder******')
        datasets_name = args.eval_dataset
        assert datasets_name in datasets_dict_3D.keys(), "The dataset use {} is not exist ".format(datasets_name)
        root = Path.db_root_dir(datasets_name)
        dataset = datasets_dict_3D[datasets_name](config=args, base_dir=root, flag=flag)
        batch_size = args.val_batch
        shuffle = False
        # num_workers = args.num_workers
        num_workers = 0
        pin_memory = False

    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            drop_last=drop_last)

    return dataset, data_loader



