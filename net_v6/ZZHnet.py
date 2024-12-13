CODE = '1120241485'
import sys
sys.path.append('/mnt/nfs/data/home/'+CODE+'/ZZHNet/')

import cv2
import torch
from torch import nn

from net_v6.encoder_mamba import encoder
from net_v6.decoder_mamba import decoder, conv_small
from net_v6.laplace import DoG


class zzh_net(nn.Module):
    def __init__(self, num_class, dims):
        super().__init__()
        self.Alap = DoG()
        self.Blap = DoG()
        self.encoder = encoder(dims)
        self.decoder = decoder(dims, num_class)
        self.clf = nn.Conv2d(dims[0], 2, 1)

    def forward(self, A, B):
        A_gray = (0.2989 * A[:, 0, :, :] + 0.5870 * A[:, 1, :, :] + 0.1140 * A[:, 2, :, :]).unsqueeze(1)
        B_gray = (0.2989 * B[:, 0, :, :] + 0.5870 * B[:, 1, :, :] + 0.1140 * B[:, 2, :, :]).unsqueeze(1)
        A_lap = self.Alap(A_gray)
        B_lap = self.Blap(B_gray)
        out = self.encoder(A, B)
        out1a, out1b, d1, out2a, out2b, d2, out3a, out3b, d3, out4a, out4b, d4 = out
        img = self.decoder(torch.cat((A_lap, B_lap), dim=1), d1, d2, d3, d4, out1a, out1b, out2a, out2b, out3a, out3b,
                           out4a, out4b)
        img = self.clf(img)

        return img

