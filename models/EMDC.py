"""
Modified from https://github.com/d-li14/mobilenetv2.pytorch 

Reference from a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import *
from .MobileNetV2 import mobilenet_v2
from .fcspn9 import AffinityPropagate1, AffinityPropagate2, AffinityPropagate3


__all__ = ['dcmodel']


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, up_scale=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch*up_scale ** 2, kernel_size=ksize, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.act = nn.PReLU(out_ch)
        self.bn = nn.BatchNorm2d(out_ch*up_scale ** 2)
        kernel = ICNR(self.conv.weight, upscale_factor=up_scale)
        self.conv.weight.data.copy_(kernel)
        self.conv.bias.data.zero_()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=ksize, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
    """Fills the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        upscale_factor: factor to increase spatial resolution by
        inizializer: inizializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = ICNR(conv_shuffle.weight, scale_factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = inizializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel


class emdc(nn.Module):
    def __init__(self, depth_norm):
        super(emdc, self).__init__()
        self.depth_norm = depth_norm

        # mobilenet_v2 as encoder
        self.enc = mobilenet_v2(True)

        # building decoder layers
        self.dec5 = UpsampleBlock(320, 160, 3)
        self.dec4 = UpsampleBlock(160+64, 96, 3)
        self.dec3 = UpsampleBlock(96+32, 64, 3)
        self.dec2 = UpsampleBlock(64+24, 32, 3)
        self.dec1 = UpsampleBlock(32+16, 24, 3)
        self.dep_conv = conv_bn_relu(24, 1, 3, 1, bn=False, relu=False)
        self.alpha = nn.Conv2d(24+32, 1, 7, stride=1, padding=3, bias=False)
        self.rezero = nn.Parameter(torch.zeros(1))
        
        self.guid_conv1_1 = conv_bn_relu(24, 9, 3, 1, bn=False, relu=False)
        self.guid_conv1_2 = conv_bn_relu(24, 9, 3, 1, bn=False, relu=False)
        self.guid_conv1_3 = conv_bn_relu(24, 9, 3, 1, bn=False, relu=False)
        self.guid_conv2_1 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv2_2 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv2_3 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv3_1 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv3_2 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv3_3 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        
        self.guid_conv4_1 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv4_2 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv5_1 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv5_2 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv6_1 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv6_2 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        
        self.guid_conv7_1 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv8_1 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        self.guid_conv9_1 = conv_bn_relu(10, 9, 3, 1, bn=False, relu=False)
        
        self.fuse_conv1 = conv_bn_relu(24, 3, 3, 1, bn=False, relu=False)
        self.fuse_conv2 = conv_bn_relu(4, 3, 3, 1, bn=False, relu=False)
        self.fuse_conv3 = conv_bn_relu(4, 3, 3, 1, bn=False, relu=False)
        
        self.fuse_conv4 = conv_bn_relu(4, 2, 3, 1, bn=False, relu=False)
        self.fuse_conv5 = conv_bn_relu(3, 2, 3, 1, bn=False, relu=False)
        self.fuse_conv6 = conv_bn_relu(3, 2, 3, 1, bn=False, relu=False)

        # building the low-level stream
        self.ls_conv0 = conv_bn_relu(2, 32, 3, 1, bn=False)
        self.ls_conv1 = conv_bn_relu(32, 32, 3, 1, bn=False)
        self.ls_conv2 = conv_bn_relu(32, 1, 3, 1, bn=False, relu=False)

        # add postprocess layer
        # total prop_time=21
        self.post_process1 = AffinityPropagate1(prop_time=3)
        self.post_process2 = AffinityPropagate1(prop_time=2)
        self.post_process3 = AffinityPropagate1(prop_time=2)
        self.post_process4 = AffinityPropagate2(prop_time=3)
        self.post_process5 = AffinityPropagate2(prop_time=2)
        self.post_process6 = AffinityPropagate2(prop_time=2)
        self.post_process7 = AffinityPropagate3(prop_time=3)
        self.post_process8 = AffinityPropagate3(prop_time=2)
        self.post_process9 = AffinityPropagate3(prop_time=2)


    def forward(self, rgb, dep):
        if self.depth_norm:
            bz = dep.shape[0]
            dep_max = torch.max(dep.view(bz,-1),1, keepdim=False)[0].view(bz,1,1,1)
            dep = dep/(dep_max+1e-4)

        x = torch.cat((rgb,dep),dim=1)
        feats = self.enc(x)

        fe7, fe4, fe3 = feats[17], feats[10], feats[6]
        fe2, fe1 = feats[3], feats[1]

        fd5 = self.dec5(fe7)
        fd4 = self.dec4(torch.cat((fd5, fe4), dim=1))
        fd3 = self.dec3(torch.cat((fd4, fe3), dim=1))
        fd2 = self.dec2(torch.cat((fd3, fe2), dim=1))
        fd1 = self.dec1(torch.cat((fd2, fe1), dim=1))
        y = self.dep_conv(fd1)
        y_glb = y

        # low-level stream
        f_ls0 = self.ls_conv0(torch.cat((dep,y), dim=1))
        f_ls1 = self.ls_conv1(f_ls0)
        f_ls2 = self.ls_conv2(f_ls1)
        y_loc = f_ls2
        fus = torch.sigmoid(self.alpha(torch.cat((f_ls1, fd1), dim=1))) * self.rezero
        y = f_ls2 * fus + y * (1 - fus)

        # fcspn postprocess
        # stage 1
        guid_map1 = self.guid_conv1_1(fd1)
        guid_map2 = self.guid_conv1_2(fd1)
        guid_map3 = self.guid_conv1_3(fd1)
        fuse1_ = self.fuse_conv1(fd1)
        fuse1 = F.softmax(fuse1_, dim=1)
        y = self.post_process1(guid_map1, guid_map2, guid_map3, fuse1, y, dep)
        
        # stage 2
        guid_map1 = self.guid_conv2_1(torch.cat((guid_map1,y), dim=1))
        guid_map2 = self.guid_conv2_2(torch.cat((guid_map2,y), dim=1))
        guid_map3 = self.guid_conv2_3(torch.cat((guid_map3,y), dim=1))
        fuse2_ = self.fuse_conv2(torch.cat((fuse1_,y), dim=1))
        fuse2 = F.softmax(fuse2_, dim=1)
        y = self.post_process2(guid_map1, guid_map2, guid_map3, fuse2, y, dep)
        
        # stage 3
        guid_map1 = self.guid_conv3_1(torch.cat((guid_map1,y), dim=1))
        guid_map2 = self.guid_conv3_2(torch.cat((guid_map2,y), dim=1))
        guid_map3 = self.guid_conv3_3(torch.cat((guid_map3,y), dim=1))
        fuse3_ = self.fuse_conv3(torch.cat((fuse2_,y), dim=1))
        fuse3 = F.softmax(fuse3_, dim=1)
        y = self.post_process3(guid_map1, guid_map2, guid_map3, fuse3, y, dep)
        
        # stage 4
        guid_map1 = self.guid_conv4_1(torch.cat((guid_map1,y), dim=1))
        guid_map2 = self.guid_conv4_2(torch.cat((guid_map2,y), dim=1))
        fuse4_ = self.fuse_conv4(torch.cat((fuse3_,y), dim=1))
        fuse4 = F.softmax(fuse4_, dim=1)
        y = self.post_process4(guid_map1, guid_map2, fuse4, y, dep)
        
        # stage 5
        guid_map1 = self.guid_conv5_1(torch.cat((guid_map1,y), dim=1))
        guid_map2 = self.guid_conv5_2(torch.cat((guid_map2,y), dim=1))
        fuse5_ = self.fuse_conv5(torch.cat((fuse4_,y), dim=1))
        fuse5 = F.softmax(fuse5_, dim=1)
        y = self.post_process5(guid_map1, guid_map2, fuse5, y, dep)
        
        # stage 6
        guid_map1 = self.guid_conv6_1(torch.cat((guid_map1,y), dim=1))
        guid_map2 = self.guid_conv6_2(torch.cat((guid_map2,y), dim=1))
        fuse6_ = self.fuse_conv6(torch.cat((fuse5_,y), dim=1))
        fuse6 = F.softmax(fuse6_, dim=1)
        y = self.post_process6(guid_map1, guid_map2, fuse6, y, dep)
        
        # stage 7
        guid_map1 = self.guid_conv7_1(torch.cat((guid_map1,y), dim=1))
        y = self.post_process7(guid_map1, y, dep)
        
        # stage 8
        guid_map1 = self.guid_conv8_1(torch.cat((guid_map1,y), dim=1))
        y = self.post_process8(guid_map1, y, dep)
        
        # stage 9
        guid_map1 = self.guid_conv9_1(torch.cat((guid_map1,y), dim=1))
        y = self.post_process9(guid_map1, y, dep)

        if self.depth_norm:
            y = y * dep_max
            y_loc = y_loc * dep_max
            y_glb = y_glb * dep_max

        return y, y_loc, y_glb


def dcmodel(depth_norm=True):
    """
    Constructs a EMDC model
    """
    return emdc(depth_norm)

