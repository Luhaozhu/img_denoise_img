""" Full assembly of the parts to form the complete network """
import time

import torch
from torch import nn

from unet_parts import DoubleConv, Down, Up, OutConv

# class PALayer(nn.Module):
#     def __init__(self, channel):
#         super(PALayer, self).__init__()
#         self.pa = nn.Sequential(
#             nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.pa(x)
#         return x * y
#
#
# class CALayer(nn.Module):
#     def __init__(self, channel):
#         super(CALayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.ca(y)
#         return x * y


class UNet(nn.Module):
    def __init__(self,bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(2, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32,64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        # self.Cmixer = CALayer(64)
        # self.Pmixer = PALayer(64)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        tempX = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return tempX[:,1:2,:,:]+logits

if __name__ == "__main__":
    import torch
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # from torch.autograd import Variable
    # from ptflops import get_model_complexity_info

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model
    torch.save(model, './unet.pth')
    # mac, params = get_model_complexity_info(model, (2,1024, 1280), as_strings=True, print_per_layer_stat=True)
    # print("%s %s" % (mac, params))
    dummy_input = torch.randn(1,2,1024,1280)
    export_onnx_file = "unet_new.onnx"
    torch_out = torch.onnx.export(model,dummy_input,export_onnx_file,
                input_names=['input'],output_names=['output'],
                dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})
