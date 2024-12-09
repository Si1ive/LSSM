import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def gauss_kernel(sigma):
    ax = np.linspace(-(5 // 2), 5 // 2, 5) #np.linspace(start, stop, num)：生成一个从 start 到 stop 的 num 个均匀分布的数值。
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

class DoG(nn.Module):
    def __init__(self):
        super(DoG, self).__init__()
        self.sigma1 = 1.0
        self.sigma2 = 2.0
        guassian_kernel1 = gauss_kernel(self.sigma1)
        guassian_kernel2 = gauss_kernel(self.sigma2)
        DoG_kernel = torch.tensor(guassian_kernel1-guassian_kernel2,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.DoG_Conv = nn.Conv2d(1,1,kernel_size=5,padding=2,bias=False)
        self.DoG_Conv.weight.data = DoG_kernel
    def forward(self, x):
        return self.DoG_Conv(x)