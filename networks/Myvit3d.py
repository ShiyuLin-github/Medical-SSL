import torch
from torch import nn

from vit_pytorch.vit_3d import ViT
from einops import rearrange

class VIT3D_RKBP(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, order_n_class=100, num_cubes=8, act='relu'):
        super(VIT3D_RKBP, self).__init__()
        
        self.vit_encoder64 = ViT(
                image_size = 110,          # image size
                frames = 72,               # number of frames
                image_patch_size = 110,     # image patch size
                frame_patch_size = 1,      # frame patch size
                num_classes = 64,          # encoded dimension
                dim = 1024,
                depth = 2,
                heads = 8,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1,
                channels = in_channels)
        # self.encoder = UNet3D_Encoder(in_channels=in_channels, act=act)
        self.gap = torch.nn.AdaptiveAvgPool3d(1)


        self.fc6 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.order_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, order_n_class)
        )

        self.ver_rot_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.hor_rot_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.mask_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.num_cubes = num_cubes
        self.sigmoid = torch.nn.Sigmoid()

    def forward_once(self, x):
        # x.shape[b, 1, 72, 110, 110]
        x_rearrange = rearrange(x, 'b c h w f -> b c f h w')
        # x_rearrange.shape[b, 1, 72, 110, 110] to fit vit

        logits = self.vit_encoder64(x_rearrange) # logits[b,64]
        return logits

    def forward(self, cubes):
        # [B, 8, C, X, Y, Z]
        cubes = cubes.transpose(0, 1)
        # [8, B, C, X, Y, Z]
        feats = []
        for i in range(self.num_cubes):
            output_fc6 = self.forward_once(cubes[i]) # output尺寸与输入一致
            feats.append(output_fc6) # 
            ## hor_rot_logit: [B, 1]

        feats = torch.cat(feats, 1) # feats[8, B, C, X, Y, Z]

        # order_logits: [B, K]
        order_logits = self.order_fc(feats)

        # hor_rot_logits: [B*8, 1]
        hor_rot_logits = self.sigmoid(self.hor_rot_fc(feats))
        ver_rot_logits = self.sigmoid(self.ver_rot_fc(feats))
        # mask
        mask_logits = self.sigmoid(self.mask_fc(feats))

        return order_logits, hor_rot_logits, ver_rot_logits, mask_logits

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        fc_layers = ['fc6', 'order_fc', 'hor_rot_fc', 'ver_rot_fc']
        module_dict = {'encoder': encoder_layers, 'fc':fc_layers}
        return module_dict
    

class VIT3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
        def __init__(self, image_size = 110,          # image size
                    frames = 72,               # number of frames
                    image_patch_size = 110,     # image patch size
                    frame_patch_size = 1,      # frame patch size
                    num_classes = 3,          # encoded dimension
                    dim = 1024,
                    depth = 2,
                    heads = 8,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    channels = 1):
            super(ViT, self).__init__()

        @staticmethod
        def get_module_dicts():
            encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
            fc_layers = ['fc6', 'order_fc', 'hor_rot_fc', 'ver_rot_fc']
            module_dict = {'encoder': encoder_layers, 'fc':fc_layers}
            return module_dict