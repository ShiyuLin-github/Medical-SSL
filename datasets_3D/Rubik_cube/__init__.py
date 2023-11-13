"""
Pretext-tasks to solve jigsaw puzzling, including Jigsaw, Rubik's cube (RKB),  Rubik's cube+ (RKB+).
"""
from datasets_3D.Rubik_cube.base_rkb_pretask import RKBBase
from datasets_3D.Rubik_cube.luna_rkb_pretask import RKBLunaPretaskSet
from datasets_3D.Rubik_cube.luna_rkb_plus_pretask import RKBPLunaPretaskSet
from datasets_3D.Rubik_cube.luna_jigsaw_pretask import JigSawLunaPretaskSet
#新增自己的数据集类
from datasets_3D.Rubik_cube.MRI_rkb_plus_pretask import RKBP_MRI_PretaskSet