import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F

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

Downsample = 'stride'


class ConvLayer(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, 3, 2)
        # self.BN=nn.BatchNorm2d(out_channels)
        # self.lu = nn.PReLU()
        #
        # self.lu=nn.ReLU()
        self.lu=Mish()
        # self.lu = nn.Hardswish()
        # self.lu = nn.Sigmoid()
        # self.lu = nn.PReLU()
        # self.lu =new()
    def forward(self, x, downsample=None):
        out = self.reflection_pad(x)
        normal = self.conv2d(out)
        # normal=self.BN(normal)
        normal = self.lu(normal)
        # normal = self.pool(normal)
        # return normal
        # normal = F.mish(normal, inplace=True)
        # normal = F.relu(normal,inplace=True)
        if downsample is"stride":
            down = self.reflection_pad(normal)
            down = self.stride_conv(down)
            # down = self.BN(down)
            # down = F.relu( down)
            down= self.lu(down)
            return normal, down
        else:
            return normal

class ConvLayer3(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(ConvLayer3, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = conv1x1(in_channels, out_channels, kernel_size=1)
        # self.BN = nn.BatchNorm2d(out_channels)
        self.lu=Mish()
        # self.lu = nn.ReLU()
        # self.lu = nn.Hardswish()
        # self.lu = nn.PReLU()
        # self.lu = nn.Sigmoid()
        # self.dropout = nn.Dropout2d(p=0.5)
        # self.lu = new()
    def forward(self, x):
        out = self.reflection_pad(x)
        normal = self.conv2d(out)
        # normal = self.BN(normal)
        # normal = F.relu(normal,inplace=True)
        normal = self.lu(normal)
        # normal = self.dropout(normal)
        return normal

# ECB :conv2、conv1
class ConvLayer1(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(ConvLayer1, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, 3, 2)
        # self.stride_conv = downconv3x3(out_channels, out_channels, kernel_size=3)
        self.dropout = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.BN=nn.BatchNorm2d(out_channels)
        # self.pool = nn.AvgPool2d(2, 2)
        self.lu = Mish()
        # self.lu = nn.ReLU()
        # self.lu = nn.Hardswish()
        # self.lu = nn.PReLU()
        # self.lu = nn.Sigmoid()
        # self.lu = new()
    def forward(self, x, downsample=None):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        # normal = self.dropout(out)
        # normal = self.BN(out)
        # normal = nn.ReLU(out)
        # normal = F.relu(out, inplace=True)
        normal = self.lu(out)
        # normal=self.pool(normal)
        # return normal
        if downsample is"stride":
            down = self.reflection_pad(normal)
            down = self.stride_conv(down)
            # down  = self.dropout(down)
            # down = self.BN(down)
            down= self.lu(down)
            return normal, down
        else:
            return normal


class ConvLayer2(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(ConvLayer2, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = conv1x1(in_channels, out_channels, kernel_size=1)
        self.lu = Mish()
        # self.lu = new()
        # self.lu = nn.PReLU()
        # self.lu = nn.Hardswish()
        # self.lu = nn.ReLU()
        # self.lu = nn.Sigmoid()
    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)
        # normal = F.relu(out, inplace=True)
        # normal = self.dropout(out)
        # out = self.BN(out)
        normal = self.lu(out)
        # normal = F.relu(out)
        return normal
#encoder
#4尺度
class EncodeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(EncodeBlock, self).__init__()
        out_channels_def = int(in_channels / 2)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv1 = ConvLayer2(in_channels, out_channels_def,1, stride)
        self.conv2 = ConvLayer1(out_channels_def, out_channels, 3, stride)

        # self.conv1 = BottleneckConvLayer(in_channels, out_channels_def, 2, stride)
        # self.conv2 = ConvLayer(out_channels_def, out_channels, kernel_size, stride)

    def forward(self, x, scales):
        normal = self.conv1(x)
        if scales == 4:
            normal = self.conv2(normal)
            return normal
        else:
            normal, out = self.conv2(normal, Downsample)
            return normal, out

#decoder
class DecodeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DecodeBlock, self).__init__()
        out_channels_def = int(in_channels / 2)

        self.conv1 = ConvLayer3(in_channels, out_channels_def, 1, stride)
        self.conv2 = ConvLayer3(out_channels_def, out_channels, 3, stride)
        # self.conv1 = conv1x1(in_channels, out_channels_def, kernel_size=1)
        # self.conv2 = conv3x3(out_channels_def, out_channels, kernel_size=3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv3(out)
        return out

# 上采样
class Upsample(torch.nn.Module):
    def __init__(self, Is_testing):
        super(Upsample, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.up = nn.ConvTranspose2d()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad = UpsampleReshape()

    def forward(self, x1, x2, Is_testing):
        out = self.up(x2)
        if Is_testing:
            out = self.pad(x1, out)
        return out


class UpsampleReshape(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape, self).__init__()

    def forward(self, shape, x):
        shape = shape.size()
        shape_x = x.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape[3] != shape_x[3]:
            lef_right = shape[3] - shape_x[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape[2] != shape_x[2]:
            top_bot = shape[2] - shape_x[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x

class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = ConvLayer(2 * channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        self.p0 = ConvLayer(2 * channels, channels, 1, 1)
        self.p1 = ConvLayer(channels, channels, 3, 1)
        self.p2 = ConvLayer(2 * channels, 2 * channels, 3, 1)
        self.p3 = ConvLayer(4 * channels, channels, 1, 1)

        block = []
        block += [ConvLayer(2 * channels, channels, 1, 1),
                  ConvLayer(channels, channels, 3, 1),
                  ConvLayer(channels, channels, 3, 1),
                  ]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        # initial layer - conv
        # print('conv')
        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init = self.conv_fusion(f_cat)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi)  # 原来的代码有问题，写成了conv_ir，现在重新训练
        out = torch.cat([out_ir, out_vi], 1)
        out1 = self.bottelblock(out)

        out2_0 = self.p0(out)
        out2_1 = self.p1(out2_0)
        out2_2 = self.p2(torch.cat([out2_0, out2_1], 1))
        out2 = self.p3(torch.cat([out2_0, out2_1, out2_2], 1))
        out = out1 + out2
        out = f_init + out
        return out


class FusionModule(nn.Module):
    def __init__(self,Is_testing,):
        super(FusionModule, self).__init__()
        # 定义一个变量
        # self.fs_type = fs_type

        # self.fusion_block1 = FusionBlock_res(nC[0], 0)
        # self.fusion_block2 = FusionBlock_res(nC[1], 1)
        # self.fusion_block3 = FusionBlock_res(nC[2], 2)
        # self.fusion_block4 = FusionBlock_res(nC[3], 3)
        rate = int(8)
        kernel_size = 3
        # self.deepsupervision = deepsupervision
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.up = nn.Upsample(scale_factor=2)
        # self.up = nn.ConvTranspose2d()

        # encoder
        # 水平方向4
        self.Conv0 = ConvLayer3(1, rate, 1, 1)
        # self.Conv0 = conv1x1(1, rate, 1)

        self.Conv1 = EncodeBlock(rate, 2*rate, kernel_size, 1)

        self.Conv2 = ConvLayer1(rate * 2, rate * 3, kernel_size, 1)
        self.ECB20 = EncodeBlock(rate*5, rate*6, kernel_size, 1)


        self.Conv3 = ConvLayer1(rate*3, rate*4, kernel_size, 1)
        self.ECB30 = EncodeBlock(rate*7, rate*8, kernel_size, 1)
        self.ECB31 = EncodeBlock(rate*18, rate*21, kernel_size, 1)

        self.Conv4 = EncodeBlock(rate*4, rate*5, kernel_size, 1)
        self.ECB40 = EncodeBlock(rate*9, rate*10, kernel_size, 1)
        self.ECB41 = EncodeBlock(rate*23, rate*24, kernel_size, 1)
        self.ECB42 = EncodeBlock(rate*60, rate*64, kernel_size, 1)

        # decoder
        self.DCB30 = DecodeBlock(rate*85, rate*16, kernel_size, 1)

        self.DCB20 = DecodeBlock(rate*27, rate*4, kernel_size, 1)
        self.DCB21 = DecodeBlock(rate*26, rate*4, kernel_size, 1)

        self.DCB10 = DecodeBlock(rate*8, rate, kernel_size, 1)
        self.DCB11 = DecodeBlock(rate*7, rate, kernel_size, 1)
        self.DCB12 = DecodeBlock(rate*8, rate, kernel_size, 1)

        self.UPf4 = Upsample(Is_testing)
        self.UPf3 = Upsample(Is_testing)
        self.UP30 = Upsample(Is_testing)
        self.UPf2 = Upsample(Is_testing)
        self.UP20 = Upsample(Is_testing)
        self.UP21 = Upsample(Is_testing)
        self.C1 = ConvLayer3(rate, 1, 1, 1)
        #
        # if self.deepsupervision:
        #     self.conv1 = ConvLayer3(rate, 1, 1, 1)
        #     self.conv2 = ConvLayer3(rate, 1, 1, 1)
        #     self.conv3 = ConvLayer3(rate, 1, 1, 1)
        # else:
        #     self.conv_out = ConvLayer3(rate, 1, 1, 1)

    def encoder(self, input):
        input=self.Conv0(input)
        # print( input.shape)
        f_conv1, d_conv1 = self.Conv1(input,1)
        # f_conv1, d_conv1 = self.Conv1(input, Downsample)

        f_conv2, d_conv2 = self.Conv2(d_conv1,Downsample)
        # f_conv2, d_conv2 = self.Conv2(d_conv1, 2)
        f_ECB20, d_ECB20 = self.ECB20(torch.cat([d_conv1, f_conv2], 1),2)

        f_conv3, d_conv3 = self.Conv3(d_conv2, Downsample)
        # f_conv3, d_conv3 = self.Conv3(d_conv2, 3)
        f_ECB30, d_ECB30 = self.ECB30(torch.cat([d_conv2, f_conv3], 1),3)
        f_ECB31, d_ECB31 = self.ECB31(torch.cat([d_ECB20, f_conv3, f_ECB30], 1),3)

        # f_conv4 = self.Conv4(d_conv3)
        f_conv4 = self.Conv4(d_conv3,4)
        f_ECB40 = self.ECB40(torch.cat([d_conv3, f_conv4], 1),4)
        f_ECB41 = self.ECB41(torch.cat([d_ECB30, f_conv4, f_ECB40], 1),4)
        f_ECB42 = self.ECB42(torch.cat([d_ECB31, f_conv4, f_ECB40, f_ECB41], 1),4)
        return [f_conv1, f_ECB20, f_ECB31, f_ECB42]
    # FS
    def fusion(self, en1, en2):
        # attention weight
        # fusion_function = fusion_strategy2.attention_fusion_weight
        # 四尺度
        f1_0 = self.fusion_block1(en1[0], en2[0])
        f2_0 = self.fusion_block2(en1[1], en2[1])
        f3_0 = self.fusion_block3(en1[2], en2[2])
        f4_0 = self.fusion_block4(en1[3], en2[3])

        return [f1_0, f2_0, f3_0, f4_0]

    def decoder(self, f_en, Is_testing):
        upf2 = self.UPf2(f_en[0], f_en[1], Is_testing)
        f_DCB10 = self.DCB10(torch.cat([f_en[0], upf2], 1))

        upf3 = self.UPf3(f_en[1], f_en[2], Is_testing)
        f_DCB20 = self.DCB20(torch.cat([f_en[1], upf3], 1))
        up20 = self.UP20(f_en[0], f_DCB20, Is_testing)
        f_DCB11 = self.DCB11(torch.cat([f_en[0], f_DCB10, up20], 1))

        up4 = self.UPf4(f_en[2], f_en[3], Is_testing)
        f_DCB30 = self.DCB30(torch.cat([f_en[2], up4], 1))
        up30 = self.UP30(f_en[1], f_DCB30, Is_testing)
        f_DCB21 = self.DCB21(torch.cat([f_en[1], f_DCB20, up30], 1))

        up21 = self.UP21(f_en[0], f_DCB21, Is_testing)
        f_DCB12 = self.DCB12(torch.cat([f_en[0], f_DCB10, f_DCB11, up21], 1))

        output = self.C1(f_DCB12)
        return output

