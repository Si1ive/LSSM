from torch import nn

from net_v1.encoder_mamba import encoder
from net_v1.d4_fuse import d4_fuse
from net_v1.decoder_mamba import decoder


class zzh_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=encoder()
        # self.d4_fuse = d4_fuse([128, 256, 512, 1024])
        # self.decoder = decoder([128, 256, 512, 1024])
        self.d4_fuse = d4_fuse([64, 128, 256, 512])
        self.decoder = decoder([64, 128, 256, 512])

        self.clf = nn.Conv2d(64,2,1)

    def forward(self, A,B):
        out = self.encoder(A,B)
        out1a, out1b, d1, out2a, out2b, d2, out3a, out3b, d3, out4a, out4b, d4 = out
        d = self.d4_fuse(d1,d2,d3,d4)
        img = self.decoder(d,out1a, out1b, out2a, out2b, out3a, out3b, out4a, out4b)
        img = self.clf(img)
        return img
