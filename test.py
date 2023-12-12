from networks.unet3d import UNet3D_RKBP
import torch

# 测试output_fc6 = self.forward_once(cubes[i])，cubes[i] = b,1,72,110,110

# 创建类的实例
my_instance = UNet3D_RKBP()

b=2
data = torch.randn([b,1,72,110,110])

output_fc6 = my_instance.forward_once(data)
print(output_fc6.shape)