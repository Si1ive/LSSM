import torch
from torch import nn

import torch.nn.functional as F

from models.vmamba2 import LayerNorm2d, VSSBlock
from net_v5.laplace import DoG

class conv_small(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(conv_small, self).__init__()
        self.ABconv= nn.Sequential(
            nn.Conv2d(inchannel, inchannel // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inchannel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel // 4, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
        )
    def forward(self, x):
        x = self.ABconv(x)
        return x

class AFF(nn.Module):
    def __init__(self, dim):
        super(AFF, self).__init__()
        self.local_att1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x1, x2):
        d = torch.cat((x1, x2), dim=1)
        d = self.local_att1(d)
        d = 1.0 + torch.tanh(d)
        d2 = torch.mul(x1, d) + torch.mul(x2, 2.0-d)
        return d2

class out(nn.Module):
    def __init__(self, dim, num_class):
        super(out, self).__init__()
        #应该先改通道还是先改尺寸呢
        self.conv = conv_small(dim,num_class)

    def upsample(self, x ,rate):
        _, _, H, W = x.size()
        return F.interpolate(x, size=(H*rate, W*rate), mode='bilinear')

    def forward(self, x, rate):
        x = self.upsample(x, rate)
        x = self.conv(x)
        return x

#编码层（即边界增强模块
class edge_block(nn.Module):
    def __init__(self, dim):
        super(edge_block, self).__init__()
        self.vssb1 = VSSBlock(
                channel_first=True,
                hidden_dim=dim,
                drop_path=0.2,
                norm_layer=LayerNorm2d,
                ssm_d_state=1,
                ssm_ratio=2.0,
                ssm_dt_rank="auto",
                ssm_act_layer=nn.SiLU,
                ssm_conv=3,
                ssm_conv_bias=False,
                ssm_drop_rate=0.0,
                ssm_init="v0",
                forward_type="v05_noz",
                mlp_ratio=4.0,
                mlp_act_layer=nn.GELU,
                mlp_drop_rate=0.0,
            )
        self.gray_conv = conv_small(dim,1)
        self.DoG = DoG()
        self.fuse = conv_small(dim*3,dim)
        self.vssb2 = VSSBlock(
                channel_first=True,
                hidden_dim=dim,
                drop_path=0.2,
                norm_layer=LayerNorm2d,
                ssm_d_state=1,
                ssm_ratio=2.0,
                ssm_dt_rank="auto",
                ssm_act_layer=nn.SiLU,
                ssm_conv=3,
                ssm_conv_bias=False,
                ssm_drop_rate=0.0,
                ssm_init="v0",
                forward_type="v05_noz",
                mlp_ratio=4.0,
                mlp_act_layer=nn.GELU,
                mlp_drop_rate=0.0,
            )
        self.aff = AFF(dim)

    def forward(self, AFFout, ABfuse, Di, Laplace):
        #主线，执行VSS
        #当AFF为None时，就是解码层第一层，主线和分支都用ABfuse,若不为None，主线用上一层的AFF结果，分支用ABfuse
        if(AFFout == None):
            AB_VSS = self.vssb1(ABfuse)
        else:
            AB_VSS = self.vssb1(AFFout)

        #增强支线
        Di = Di * AB_VSS
        # 执行laplace之前要先转换成灰度图
        ABfuse = self.DoG(self.gray_conv(ABfuse)) * AB_VSS
        Laplace= Laplace * AB_VSS
        c = self.fuse(torch.cat((Di, ABfuse,Laplace), dim=1))
        c = self.vssb2(c)

        #主线支线融合
        out = self.aff(AB_VSS,c)

        return out


class decoder(nn.Module):
    def __init__(self, dim, num_class):
        super(decoder, self).__init__()
        #编码第一层
        self.ABfuse1 = conv_small(dim[3]*2,dim[3])
        self.de_block1 = edge_block(dim[3])
        #当前层输出图像
        #self.out1 = out(dim[3],num_class)
        self.conv1 = conv_small(dim[3],dim[2])

        # 编码第二层
        self.ABfuse2 = conv_small(dim[2]*2,dim[2])
        self.de_block2 = edge_block(dim[2])
        #self.out2 = out(dim[2],num_class)
        self.conv2 = conv_small(dim[2],dim[1])

        # 编码第三层
        self.ABfuse3 = conv_small(dim[1]*2,dim[1])
        self.de_block3 = edge_block(dim[1])
        #self.out3 = out(dim[1],num_class)
        self.conv3 = conv_small(dim[1],dim[0])

        # 编码第四层
        self.ABfuse4 = conv_small(dim[0]*2,dim[0])
        self.de_block4 = edge_block(dim[0])
        self.out4 = out(dim[0],num_class)

    def upsample(self, x ,rate):
        _, _, H, W = x.size()
        return F.interpolate(x, size=(H*rate, W*rate), mode='bilinear')

    def downsample(self, x ,rate):
        _, _, H, W = x.size()
        return F.interpolate(x, size=(H // rate, W // rate), mode='bilinear')

    def forward(self, lap, d1, d2, d3, d4, out1a, out1b, out2a, out2b, out3a, out3b, out4a, out4b):
        #解码第一层
        #融合一下编码层
        ABfuse1 = self.ABfuse1(torch.cat((out4a, out4b),dim=1))
        #laplace要降低2*2*2*4 32倍
        lap1 = self.downsample(lap, 32)
        #执行编码层核心
        c1 = self.de_block1(None, ABfuse1, d4, lap1)
        #输出当前层图像
        #out1 = self.out1(c1,32)
        # 提升一下尺寸
        c1 = self.upsample(c1, 2)
        #降低一下通道
        c1 = self.conv1(c1)


        # 解码第二层
        #差一个调整lap大小,差异和编码层输出特征都是调好的所以不用动
        ABfuse2 = self.ABfuse2(torch.cat((out3a, out3b),dim=1))
        lap2 = self.downsample(lap, 16)
        c2 = self.de_block2(c1, ABfuse2, d3, lap2)
        #out2 = self.out2(c2,16)
        #out输出当前层，c是继续往后走
        c2 = self.upsample(c2, 2)
        c2 = self.conv2(c2)


        # 解码第三层
        ABfuse3 = self.ABfuse3(torch.cat((out2a, out2b),dim=1))
        lap3 = self.downsample(lap, 8)
        c3 = self.de_block3(c2, ABfuse3, d2, lap3)
        #out3= self.out3(c2,8)
        c3 = self.upsample(c3, 2)
        c3 = self.conv3(c3)


        # 解码第四层
        ABfuse4 = self.ABfuse4(torch.cat((out1a, out1b),dim=1))
        lap4 = self.downsample(lap, 4)
        c4 = self.de_block4(c3, ABfuse4, d1, lap4)
        out4 = self.upsample(c4,4)

        #return out1,out2,out3,out4
        return out4