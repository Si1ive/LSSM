import torch
from torch import nn

class d4_fuse(nn.Module):
    def __init__(self,dim):
        super(d4_fuse, self).__init__()
        #二合一，通道恢复
        self.upsample1 = nn.Conv2d(dim[0], dim[1], 3,stride=2, padding=1, bias=False)
        self.local_att1 = nn.Sequential(
            nn.Conv2d(dim[1] * 2, dim[1] // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim[1] // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim[1] // 4, dim[1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim[1]),
        )

        self.upsample2 = nn.Conv2d(dim[1], dim[2], 3,stride=2, padding=1, bias=False)
        self.local_att2 = nn.Sequential(
            nn.Conv2d(dim[2] * 2, dim[2] // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim[2] // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim[2] // 4, dim[2], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim[2]),
        )

        self.upsample3 = nn.Conv2d(dim[2], dim[3], 3,stride=2, padding=1, bias=False)
        self.local_att3 = nn.Sequential(
            nn.Conv2d(dim[3] * 2, dim[3] // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim[3] // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim[3] // 4, dim[3], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim[3]),
        )

    def forward(self,d1,d2,d3,d4):
        #四个差异特征，进行三次融合
        #dims=[96, 192, 384, 768],
        #尺寸每层少一半 256 128 64 32
        #将大尺寸小通道特征 往 小尺寸大通道上靠
    #第一轮融合
        d1 = self.upsample1(d1)
        d = torch.cat((d1, d2), dim=1)
        d = self.local_att1(d)
        d = 1.0 + torch.tanh(d)
        d2 = torch.mul(d1, d) + torch.mul(d2, 2.0-d)
    #第二轮融合
        d2 = self.upsample2(d2)
        d = torch.cat((d2, d3), dim=1)
        d = self.local_att2(d)
        d = 1.0 + torch.tanh(d)
        d3 = torch.mul(d2, d) + torch.mul(d3, 2.0 - d)
    #第三轮融合
        d3 = self.upsample3(d3)
        d = torch.cat((d3, d4), dim=1)
        d = self.local_att3(d)
        d = 1.0 + torch.tanh(d)
        d4 = torch.mul(d3, d) + torch.mul(d4, 2.0 - d)
        return d4