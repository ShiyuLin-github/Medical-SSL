import torch
from torch import nn

from vit_pytorch.vit_3d import ViT
from vit_pytorch import SimpleViT

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        
    


class oneVIT3D_RKBP(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, order_n_class=100, num_cubes=8, act='relu'):
        super(oneVIT3D_RKBP, self).__init__()
        
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

def pair(t):
    return t if isinstance(t, tuple) else (t, t) #pair函数：这是一个辅助函数，用于从一个数值或元组中提取两个值。它将确保给定的值被分配给两个不同的变量，通常用于处理图像尺寸和patch尺寸的输入。其作用是将输入 t 转换为一个元组，如果 t 已经是元组，则保持不变，否则创建一个包含两个相同元素的元组 (t, t)。

# classes

class FeedForward(nn.Module): #FeedForward类：这是一个前馈神经网络模块，用于对输入进行线性变换、GELU激活函数和dropout操作。它通常在Transformer的每个层中用于处理特征。
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module): #这是注意力机制模块，用于计算注意力分布。它包括对输入进行Layer Normalization、计算注意力分布、应用dropout和输出处理。注意力机制通常用于捕获输入序列中不同位置的依赖关系。
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim) #project_out：一个布尔值，指示是否需要将注意力输出映射回原始维度，当heads=1且dim_head=dim时返回true，not一下返回false，其他情况都是true

        self.heads = heads
        self.scale = dim_head ** -0.5 #理论部分的除权数，通过缩放因子，确保注意力分布的尺度适当，从这个角度考虑dim_head是K的维度

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1) #dim=-1意味着对最后一个维度进行softmax计算
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() #nn.Identity() 不对输入进行任何变换，只是将输入复制到输出。

    def forward(self, x):
        x = self.norm(x) # x[b, n+1, dim] [4, 309, 1024]
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 通过 chunk(3, dim=-1) 将 qkv 分成三部分。q是[b, 309, 512]其中512=dim_heads(64) * heads(8),k与v同，qkv是len=3的tensor里面包含q,k,v

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) #map 是一个Python内置函数，它的主要作用是将一个函数应用于可迭代对象（如列表、元组等），并返回一个包含函数应用结果的新可迭代对象。map(function here, iterable here); lambda [arg1 [,arg2,.....argn]]:expression
        # 先是用lambda函数建了个匿名函数:对t进行rearrange, 再将lambda用map套用到qkv上，将每个qkv变为4维
        # q[b, 8, 309, 64] 其中8=heads，64=head_dim 又把融合在一起的矩阵给拆开

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # 矩阵转置的意义在于把i变为j，j变为i，这里本可以直接.T的，但应该是为了保险设置为让矩阵的最后一维和倒数第二个维度进行交换，如果是4维的矩阵，这里定义转置的维度是有意义的所有不直接.T也有道理
        # dots[b, heads, n+1, n+1]  [b, 8, 309, 309],最后的[309,309]代表每个patch与每个patch之间的关系矩阵

        attn = self.attend(dots) # attn[b, heads, n+1, n+1] shape同上，进行了一次softmax归一化
        attn = self.dropout(attn) # same as before

        out = torch.matmul(attn, v) # out[b, heads, n+1, dim_head] [4, 8, 309, 64] 对最后的矩阵计算相乘相加的点乘操作得出注意力分数维度为dim_head
        out = rearrange(out, 'b h n d -> b n (h d)') # 转回之前的尺寸方便进行下一步循环 #[4, 309, 512]
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x # 设置循环层数的迭代，这里的累加是为了达到类似resnet的效果

class My3dViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, order_n_class, num_cubes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) #pair 函数用于从一个给定的数值或元组中提取两个值，并将它们分配给两个不同的变量。在上面的代码片段中，pair 函数被用来从图像尺寸和图像patch尺寸中提取高度和宽度。
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size' #确保尺寸能被整除

        # num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        num_patches = frames
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.fc6 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.order_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, order_n_class)
        )

        self.ver_rot_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.hor_rot_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.mask_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        # self.num_cubes = num_cubes
        self.sigmoid = torch.nn.Sigmoid()

        self.twodvit = SimpleViT(
                    image_size = 240,
                    patch_size = 16,
                    num_classes = dim,
                    dim = 1024,
                    depth = 1,
                    heads = 1,
                    mlp_dim = 2048,
                    channels = 1
                    )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c d h w -> (b d) c h w'),
            SimpleViT(
                    image_size = 240,
                    patch_size = 120,
                    num_classes = dim,
                    dim = 1024,
                    depth = 1,
                    heads = 1,
                    mlp_dim = 2048
                    ),
            nn.LayerNorm(dim), 
            Rearrange('(b d) dim -> b d dim',d = frames),
        )
        self.torearrange = Rearrange('b c h w d -> (b d) c h w')
        self.torearrange_bck = Rearrange('(b d) dim -> b d dim',d = frames)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) #nn.Parameter 允许将张量标记为模型的参数，使其能够参与训练，并在反向传播时进行更新。它的形状是 (1, num_patches + 1, dim)，其中 num_patches 表示输入图像被划分成的图像块的数量，加 1 是为了算上cls_tokens(参下方代码)。dim 是 patch_embedding 的dimension，最后patch与pos相加以此保留位置信息。

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) #self.cls_token 也是一个可学习的参数，用于表示一个额外的类别标记。它的形状是 (1, 1, dim)，其中 dim 是Transformer的输入和输出维度。这个类别标记通常与图像块的表示连接在一起，以捕获全局信息。
        self.dropout = nn.Dropout(emb_dropout) #self.dropout 是一个Dropout层，用于在输入嵌入上应用丢弃操作。丢弃操作有助于防止过拟合，它以概率 emb_dropout 随机将输入中的一些元素设置为零。

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, video):
        # x = self.to_patch_embedding(video) # video[b,c,154,240,240]--> x[b, 308, 1024]' dim=1024, 308是n也就是一共有多少个patch
        # video[4,1,240,240,154]
        x = self.torearrange(video) # x[616, 1, 240, 240] (b*d, c, h, w)
        x = self.twodvit(x) # x[616, 1024] (b*d, dim)
        x = self.torearrange_bck(x) # x[4, 154, 1024]

        b, n, _ = x.shape # b n dim

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # cls_token[b, 1, dim=1024]  用repeat函数复制cls_tokens的维度为b 1 d，反正也是随机的，为啥一开始不直接这样，难道是因为不知道 batch_size?
        x = torch.cat((cls_tokens, x), dim=1) # x.shape从[b,n,-]变为[b,n+1,-],[b,308,1024]-->[b,309,1024]
        x += self.pos_embedding[:, :(n + 1)] # pos_embedding[1, 309, 1024],后面这个[:, :(n + 1)]应该只是为了不出bug,+=的时候用到了广播机制扩展到batch_size的维度上
        x = self.dropout(x) # same as before

        x = self.transformer(x) # same as before [b, n+1, dim] [4, 309, 1024]

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # [b, dim] [b, 1024] 如果mean则取tranformer的均值，否则取第一个

        # x = self.to_latent(x) # 不清楚作用
        # return self.mlp_head(x)

        # order_logits: [B, K]
        order_logits = self.order_fc(x)

        # hor_rot_logits: [B*8, 1]
        hor_rot_logits = self.sigmoid(self.hor_rot_fc(x))
        ver_rot_logits = self.sigmoid(self.ver_rot_fc(x))
        # mask
        mask_logits = self.sigmoid(self.mask_fc(x))

        return order_logits, hor_rot_logits, ver_rot_logits, mask_logits    
