import torch
from torch import nn

from net_v3.encoder_mamba import encoder
from net_v3.decoder_mamba import decoder, conv_small
from net_v3.laplace import DoG


class zzh_net(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.lap = DoG()
        self.ABfuse = conv_small(6,1)
        self.encoder=encoder()
        self.decoder = decoder([64, 128, 256, 512],num_class)

    def forward(self, A,B):
        lap = self.lap(self.ABfuse(torch.cat((A,B),dim=1)))
        out = self.encoder(A,B)
        out1a, out1b, d1, out2a, out2b, d2, out3a, out3b, d3, out4a, out4b, d4 = out
        img1, img2, img3, img4 = self.decoder(lap, d1, d2, d3, d4, out1a, out1b, out2a, out2b, out3a, out3b, out4a, out4b)
        return img1, img2, img3, img4
