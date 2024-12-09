import torch
from torch import nn

import torch.nn.functional as F
from models.vmamba2 import LayerNorm2d, VSSBlock, Permute

class resnet(nn.Module):
    def __init__(self,dim1,dim2):
        super(resnet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.seq(x)
        return x

class decoder(nn.Module):
    def __init__(self,dim):
        super(decoder, self).__init__()
        #三个特征逐通道组合
        self.de_block1 = nn.Sequential(
            nn.Conv2d(dim[3]*3, dim[3], kernel_size=1),
            VSSBlock(
                channel_first=True,
                hidden_dim=dim[3],
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
            ),
        )
        self.de_block2 = nn.Sequential(
            nn.Conv2d(dim[2]*3, dim[2], kernel_size=1),
            VSSBlock(
                channel_first=True,
                hidden_dim=dim[2],
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
            ),
        )
        self.de_block3 = nn.Sequential(
            nn.Conv2d(dim[1]*3, dim[1], kernel_size=1),
            VSSBlock(
                channel_first=True,
                hidden_dim=dim[1],
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
            ),
        )
        self.de_block4 = nn.Sequential(
            nn.Conv2d(dim[0]*3, dim[0], kernel_size=1),
            VSSBlock(
                channel_first=True,
                hidden_dim=dim[0],
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
            ),
        )
        self.res1 = resnet(dim[3], dim[2])
        self.res2 = resnet(dim[2], dim[1])
        self.res3 = resnet(dim[1], dim[0])

    def upsample(self, x ,rate):
        _, _, H, W = x.size()
        return F.interpolate(x, size=(H*rate, W*rate), mode='bilinear')

    def forward(self, d,out1a, out1b, out2a, out2b, out3a, out3b, out4a, out4b):
        #上采样呢？
        #相当于是第二层融合时应该有问题

        #解码第一层
        c1 = torch.cat([d, out4a, out4b],dim=1)
        c1 = self.de_block1(c1)
        c1 = self.res1(c1)
        c1 = self.upsample(c1,2)
        # 解码第二层
        c2 = torch.cat([c1, out3a, out3b], dim=1)
        c2 = self.de_block2(c2)
        c2 = self.res2(c2)
        c2 = self.upsample(c2,2)
        # 解码第三层
        c3 = torch.cat([c2, out2a, out2b], dim=1)
        c3 = self.de_block3(c3)
        c3 = self.res3(c3)
        c3 = self.upsample(c3,2)
        # 解码第四层
        c4 = torch.cat([c3, out1a, out1b], dim=1)
        c4 = self.de_block4(c4)
        c4 = self.upsample(c4,4)

        return c4