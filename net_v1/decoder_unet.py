import torch
from torch import nn

from models.vmamba2 import LayerNorm2d

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm2d(ch_out),
            nn.GELU(),
        )
    def forward(self, x):
        x = x + self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm2d(ch_out),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class BN_GE(nn.Module):
    def __init__(self, ch_out):
        super(BN_GE, self).__init__()
        self.BN_GE = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.GELU(inplace=True)
        )

    def forward(self, x):
        x = self.BN_GE(x)
        return x

class decoder(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, patch_size=256):
        super(decoder, self).__init__()
        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv6 = conv_block(ch_in=256, ch_out=256)
        self.Conv_1x1_6 = nn.Conv2d(256, output_ch, kernel_size=1, stride=1, padding=0)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)
        self.Conv_1x1_5 = nn.Conv2d(128, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1_4 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)
        #边界卷积根本没什么不一样啊，怎么提取的边界信息？
        #单纯是通过损失函数来进行约束，然后使得这个卷积具有这样的性质吗？
        #所以这也就串联起来了为什么要把边界约束也作为结果进行损失的计算
        #那为什么dout要纳入损失计算呢？
        self.Conv_1x1_3_edge = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_3_edge_ = BN_GE(ch_out=output_ch)
        self.Conv_1x1_3 = nn.Conv2d(34, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)
        self.Conv_1x1_2_edge = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(18, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, d,out1,out2,out3,out4):
        #纯两层卷积一下，啥也没变
        x5 = self.Up_conv6(x5)
        #为什么d6不能用，非得用x5？
        #用的是d几，而不是dout
        #1*1这种卷积都是为了缩小通道到二通道，使得当前的d直接得到一张图像数据
        d6_out = self.Conv_1x1_6(x5)
        #scale_factor=16将长宽扩大16倍

        #Up都是尺寸扩大两倍
        x5 = self.Up5(x5)
        d5 = torch.cat((x5, x4), dim=1)
        #缩小通道2倍
        d5 = self.Up_conv5(d5)
        d5_out = self.Conv_1x1_5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)
        d4_out = self.Conv_1x1_4(d4)


        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)
        d3_out = self.Conv_1x1_3(d3)


        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)
        d2_out = self.Conv_1x1_2(d2)
        return d6_out, d5_out, d4_out, d3_out, d2_out, d3_edge, d2_edge