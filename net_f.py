import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-10
class new(nn.Module):
    def __init__(self,inplace=True):
        super().__init__()

    def forward(self, x):
        return (F.softplus(2*x)-0.6931471805599453)/2  # ln2

def lncosh(x):
    ln2 = 0.6931471805599453
    return (2*x+torch.log(1+torch.exp(-2*x))-ln2)/2

def ori(x):
    return torch.log(torch.exp(x)*torch.cosh(x))/2



class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # softplus(x)：ln(mri+exp**x)
        # mri、无上限，但是有下限；2、光滑；M-SPE、非单调
        return x *(torch.tanh(F.softplus(x)))

def var(x, dim=0):
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


class MultConst(nn.Module):
    def forward(self, input):
        return 255*input


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[1] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        # 空洞卷积
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last
        # self.lu = nn.ReLU()

        # self.lu = nn.Hardswish()
        # self.lu = nn.Sigmoid()
        # self.lu = nn.PReLU()
        self.lu = Mish()
        # self.lu = new()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.lu(out)
        #     # out = self.dropout(out)
        return out
# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16.png
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res, self).__init__()
        ws = [3, 3, 3, 3]

        self.conv_fusion = ConvLayer(2*channels, channels,ws[index], 1)
        # self.conv_fusion1 = ConvLayer( channels, channels, 3, 1)
        # self.conv_fusion2 = ConvLayer( channels, channels, 3, 1)


        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        # 设置卷积核
        self.p0=ConvLayer(2*channels, channels, 1, 1)
        self.p1=ConvLayer(channels, channels, 3, 1)
        self.p2 =ConvLayer(2*channels, 2*channels, 3, 1)
        self.p3= ConvLayer(4 * channels, channels, 1, 1)

        # self.p1 = ConvLayer(channels, channels, 3, 1)
        # self.p2 = ConvLayer (channels, channels, 3, 1)
        # self.p3 = ConvLayer(channels, channels, 1, 1)

        block = []
        block += [ConvLayer(2 * channels, channels, 1, 1),
                  ConvLayer(channels, channels, 3, 1),
                  ConvLayer(channels, channels,3, 1),
                  ]
        self.bottelblock = nn.Sequential(*block)



    def forward(self, x_ir, x_vi):
        # initial layer - conv
        # print('conv')
        f_cat = torch.cat([x_ir, x_vi], 1)
        # f_init = self.conv_fusion0(f_cat)
        f_init = self.conv_fusion(f_cat)
        # f_init = self.conv_fusion1(f_init)
        # f_init = self.conv_fusion2(f_init)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi) # 原来的代码有问题，写成了conv_ir，现在重新训练
        out = torch.cat([out_ir, out_vi], 1)
        out1 = self.bottelblock(out)

        out2_0=self.p0(out)
        out2_1 = self.p1(out2_0)
        out2_2 = self.p2(torch.cat([out2_0,out2_1],1))
        out2= self.p3(torch.cat([out2_0,out2_1,out2_2],1))

        # out2_0 = self.p0(out)
        # out2_1 = self.p1(out2_0)
        # out2_2 = self.p2(out2_1)
        # out2 = self.p3(out2_2)

        out = out1 + out2
        out = f_init + out
        return out


# Fusion network, 4 groups of features
# 自己设计的融合层（low_RFN）
class Fusion_network(nn.Module):
    def __init__(self, nC, fs_type):
        super(Fusion_network, self).__init__()
        self.fs_type = fs_type

        self.fusion_block1 = FusionBlock_res(nC[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]


