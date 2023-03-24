#!/usr/bin/env python3
import torch
import torch.nn as nn
from collections import OrderedDict

depth_wise=False
def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        modules['pointwise'] = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_relu:
        modules['relu'] = nn.ReLU()

    return nn.Sequential(modules)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=5, stride=stride, padding=2, is_seperable=depth_wise, has_relu=True)
        self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=depth_wise, has_relu=False)

        self.proj = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels else
            Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=depth_wise, has_relu=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        proj = self.proj(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + proj
        return self.relu(x)


def EncoderStage(in_channels: int, out_channels: int, num_blocks: int):

    blocks = [
        EncoderBlock(
            in_channels=in_channels,
            mid_channels=out_channels//4,
            out_channels=out_channels,
            stride=2,
        )
    ]
    for _ in range(num_blocks-1):
        blocks.append(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels//4,
                out_channels=out_channels,
                stride=1,
            )
        )

    return nn.Sequential(*blocks)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2
        self.conv0 = Conv2D(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=depth_wise, has_relu=True,
        )
        self.conv1 = Conv2D(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=depth_wise, has_relu=False,
        )

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = x + inp
        return x


class DecoderStage(nn.Module):

    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int):
        super().__init__()

        self.decode_conv = DecoderBlock(in_channels, in_channels, kernel_size=3)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=1, stride=1, padding=0, is_seperable=depth_wise, has_relu=True)
        # M.init.msra_normal_(self.upsample.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, inputs):
        inp, skip = inputs

        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return x + y

# from torch.nn.init import xavier_uniform_, constant_
# import numpy as np
# from ops.modules.ms_deform_attn import MSDeformAttn, MSDeformAttn_Fusion
# class DeformableAttnBlock(nn.Module):
#     def __init__(self, n_heads=4, n_levels=3, n_points=4, d_model=32):
#         super().__init__()
#         self.n_levels = n_levels
#
#         self.defor_attn = MSDeformAttn(d_model=d_model, n_levels=3, n_heads=n_heads, n_points=n_points)
#         self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
#         self.emb_qk = nn.Conv2d(3 * d_model + 8, 3 * d_model, kernel_size=3, padding=1)
#         self.emb_v = nn.Conv2d(3 * d_model, 3 * d_model, kernel_size=3, padding=1)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#
#         self.feedforward = nn.Sequential(
#             nn.Conv2d(2 * d_model, d_model, kernel_size=3, padding=2, dilation=2),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
#         )
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#
#     def get_reference_points(self, spatial_shapes, valid_ratios, device):
#         reference_points_list = []
#         for lvl, (H_, W_) in enumerate(spatial_shapes):
#             ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
#                                           torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
#             ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
#             ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points
#
#     def get_valid_ratio(self, mask):
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio
#
#     def preprocess(self, srcs):
#         bs, t, c, h, w = srcs.shape
#         masks = [torch.zeros((bs, h, w)).bool().to(srcs.device) for _ in range(t)]
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
#         src_flatten = []
#         mask_flatten = []
#         spatial_shapes = []
#         for lv1 in range(t):
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
#         return spatial_shapes, valid_ratios
#
#     def forward(self, frame, srcframe, flow_forward, flow_backward):
#         b, t, c, h, w = frame.shape
#         # bs,t,c,h,w = frame.shape
#         warp_fea01 = warp(frame[:, 0], flow_backward[:, 0])
#         warp_fea21 = warp(frame[:, 2], flow_forward[:, 1])
#
#         qureys = self.act(self.emb_qk(torch.cat([warp_fea01, frame[:, 1], warp_fea21, flow_forward.reshape(b, -1, h, w),
#                                                  flow_backward.reshape(b, -1, h, w)], 1))).reshape(b, t, c, h, w)
#
#         value = self.act(self.emb_v(frame.reshape(b, t * c, h, w)).reshape(b, t, c, h, w))
#
#         spatial_shapes, valid_ratios = self.preprocess(value)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=value.device)
#
#         output = self.defor_attn(qureys, reference_points, value, spatial_shapes, level_start_index, None, flow_forward,
#                                  flow_backward)
#
#         output = self.feed_forward(output)
#         output = output.reshape(b, t, c, h, w) + frame
#
#         tseq_encoder_0 = torch.cat([output.reshape(b * t, c, h, w), srcframe.reshape(b * t, c, h, w)], 1)
#         output = output.reshape(b * t, c, h, w) + self.feedforward(tseq_encoder_0)
#         return output.reshape(b, t, c, h, w), srcframe
#
#
# class DeformableAttnBlock_FUSION(nn.Module):
#     def __init__(self, n_heads=4, n_levels=3, n_points=4, d_model=32):
#         super().__init__()
#         self.n_levels = n_levels
#
#         self.defor_attn = MSDeformAttn_Fusion(d_model=d_model, n_levels=3, n_heads=n_heads, n_points=n_points)
#         self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
#         self.emb_qk = nn.Conv2d(3 * d_model + 4, 3 * d_model, kernel_size=3, padding=1)
#         self.emb_v = nn.Conv2d(3 * d_model, 3 * d_model, kernel_size=3, padding=1)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#
#         self.feedforward = nn.Sequential(
#             nn.Conv2d(2 * d_model, d_model, kernel_size=3, padding=2, dilation=2),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
#         )
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#         self.fusion = nn.Sequential(
#             nn.Conv2d(d_model, d_model, kernel_size=1, padding=0),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#
#     def get_reference_points(self, spatial_shapes, valid_ratios, device):
#         reference_points_list = []
#         for lvl, (H_, W_) in enumerate(spatial_shapes):
#             ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
#                                           torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
#             ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
#             ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points
#
#     def get_valid_ratio(self, mask):
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio
#
#     def preprocess(self, srcs):
#         bs, t, c, h, w = srcs.shape
#         masks = [torch.zeros((bs, h, w)).bool().to(srcs.device) for _ in range(t)]
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
#         src_flatten = []
#         mask_flatten = []
#         spatial_shapes = []
#         for lv1 in range(t):
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
#         return spatial_shapes, valid_ratios
#
#     def forward(self, frame, srcframe, flow_forward, flow_backward):
#         b, t, c, h, w = frame.shape
#         # bs,t,c,h,w = frame.shape
#         warp_fea01 = warp(frame[:, 0], flow_backward[:, 0])
#         warp_fea21 = warp(frame[:, 2], flow_forward[:, 1])
#
#         qureys = self.act(self.emb_qk(
#             torch.cat([warp_fea01, frame[:, 1], warp_fea21, flow_forward[:, 1], flow_backward[:, 0]], 1))).reshape(b, t,
#                                                                                                                    c, h,
#                                                                                                                    w)
#
#         value = self.act(self.emb_v(frame.reshape(b, t * c, h, w)).reshape(b, t, c, h, w))
#
#         spatial_shapes, valid_ratios = self.preprocess(value)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         reference_points = self.get_reference_points(spatial_shapes[0].reshape(1, 2), valid_ratios, device=value.device)
#
#         output = self.defor_attn(qureys, reference_points, value, spatial_shapes, level_start_index, None, flow_forward,
#                                  flow_backward)
#
#         output = self.feed_forward(output)
#         output = output.reshape(b, c, h, w) + frame[:, 1]
#
#         tseq_encoder_0 = torch.cat([output, srcframe[:, 1]], 1)
#         output = output.reshape(b, c, h, w) + self.feedforward(tseq_encoder_0)
#         output = self.fusion(output)
#         return output
#
#
# def warp(x, flo):
#     """
#     warp an image/tensor (im2) back to im1, according to the optical flow
#         x: [B, C, H, W] (im2)
#         flo: [B, 2, H, W] flow
#     """
#     B, C, H, W = x.size()
#     # mesh grid
#     xx = torch.arange(0, W).reshape(1, -1).repeat(H, 1)
#     yy = torch.arange(0, H).reshape(-1, 1).repeat(1, W)
#     xx = xx.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
#     yy = yy.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
#     grid = torch.cat((xx, yy), 1).float()
#     grid = grid.to(x.device)
#     vgrid = Variable(grid) + flo
#
#     # scale grid to [-1,1]
#     vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
#     vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
#
#     vgrid = vgrid.permute(0, 2, 3, 1)
#     output = nn.functional.grid_sample(x, vgrid, padding_mode='border', align_corners=True)
#     # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
#     # mask = nn.functional.grid_sample(mask, vgrid,align_corners=True )
#
#     # mask[mask < 0.999] = 0
#     # mask[mask > 0] = 1
#
#     # output = output * mask
#
#     return output
class Network(nn.Module):

    def __init__(self, n_feat=32):
        super().__init__()

        self.conv0 = Conv2D(in_channels=2, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=16, out_channels=64, num_blocks=2)
        self.enc2 = EncoderStage(in_channels=64, out_channels=128, num_blocks=2)
        self.enc3 = EncoderStage(in_channels=128, out_channels=256, num_blocks=4)
        self.enc4 = EncoderStage(in_channels=256, out_channels=512, num_blocks=4)

        self.encdec = Conv2D(in_channels=512, out_channels=64, kernel_size=3, padding=1, stride=1, is_seperable=depth_wise, has_relu=True)
        self.dec1 = DecoderStage(in_channels=64, skip_in_channels=256, out_channels=64)
        self.dec2 = DecoderStage(in_channels=64, skip_in_channels=128, out_channels=32)
        self.dec3 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32)
        self.dec4 = DecoderStage(in_channels=32, skip_in_channels=16, out_channels=16)

        self.out0 = DecoderBlock(in_channels=16, out_channels=16, kernel_size=3)
        self.out1 = Conv2D(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        #############################
        # self.MMA = DeformableAttnBlock(n_heads=4,d_model=128,n_levels=3,n_points=12)
        # # self.Defattn2 = DeformableAttnBlock(n_heads=8,d_model=128,n_levels=3,n_points=12)
        # self.MSA = DeformableAttnBlock_FUSION(n_heads=4,d_model=128,n_levels=3,n_points=12)
        # self.motion_branch = torch.nn.Sequential(
        #             torch.nn.Conv2d(in_channels=2*n_feat * 4, out_channels=96//2, kernel_size=3, stride=1, padding=8, dilation=8),
        #             nn.LeakyReLU(0.1,inplace=True),
        #             torch.nn.Conv2d(in_channels=96//2, out_channels=64//2, kernel_size=3, stride=1, padding=16, dilation=16),
        #             nn.LeakyReLU(0.1,inplace=True),
        #             torch.nn.Conv2d(in_channels=64//2, out_channels=32//2, kernel_size=3, stride=1, padding=1, dilation=1),
        #             nn.LeakyReLU(0.1,inplace=True),
        # )
        # self.motion_out = torch.nn.Conv2d(in_channels=32//2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        # constant_(self.motion_out.weight.data, 0.)
        # constant_(self.motion_out.bias.data, 0.)
    def forward(self, inp):

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        up3 = self.dec1((conv5, conv3))
        up2 = self.dec2((up3, conv2))
        up1 = self.dec3((up2, conv1))
        x = self.dec4((up1, conv0))

        x = self.out0(x)
        x = self.out1(x)
        # pred = inp[:, 0:1, :, :] + x
        pred = inp[:,1:2,:,:] + x
        # pred = inp[:, 2:3, :, :] + x
        # pred = inp + x
        return pred

class model_L(nn.Module):
    def __init__(self):
        super(model_L, self).__init__()
        checkpoint = torch.load('/data/aaron/quantization_deploy/img_denoise/model/16_Epoch5750-Total_Loss0.0084.pth')
        self.net = Network()
        self.net.load_state_dict(checkpoint.state_dict())

    def forward(self, inp):
        x=inp[:,0:2,:,:]
        out = self.net(x)
        return out


    # 分析FLOPs
    # flops = FlopCountAnalysis(model, tensor)
    # print("FLOPs: ", flops.total()/1e9)

    # 分析parameters
    # print(parameter_count_table(model))




