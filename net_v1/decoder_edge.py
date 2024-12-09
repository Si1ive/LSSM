import torch
from torch import nn

class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0,groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))


    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()

        self.de_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.de_block2 = DWConv(out_channels, out_channels)

        self.de_block3 = DWConv(out_channels, out_channels)

        self.de_block4 = nn.Conv2d(out_channels, 1, 1)

        self.de_block5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, input1, input, input2):
        x0 = torch.cat((input1, input, input2), dim=1)
        x0 = self.de_block1(x0)
        x = self.de_block2(x0)
        x = self.de_block3(x)
        x = x + x0
        al = self.de_block4(x)
        result = self.de_block5(x)
        return al, result


class ref_seg(nn.Module):
    def __init__(self):
        super(ref_seg, self).__init__()
        self.dir_head = nn.Sequential(nn.Conv2d(32, 32, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 8, 1, 1))
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1, bias=False)
        self.conv0.weight = nn.Parameter(torch.tensor([[[[0, 0, 0], [1, 0, 0], [0, 0, 0]]],
                                                       [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 1, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 1, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]]).float())

    def forward(self, x, masks_pred, edge_pred):
        #edge_pred1 x2 masks_pred3
        direc_pred = self.dir_head(x)
        direc_pred = direc_pred.softmax(1)
        #torch.sigmoid(edge_pred).detach() > 0.5 得到的是布尔值，跟1乘转换成01
        edge_mask = 1 * (torch.sigmoid(edge_pred).detach() > 0.5)
        refined_mask_pred = (self.conv0(masks_pred) * direc_pred).sum(1).unsqueeze(1) * edge_mask + masks_pred * (
                    1 - edge_mask)
        return refined_mask_pred


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.db1 = nn.Sequential(
            nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512), nn.ReLU(),
            DWConv(512, 512),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )

        self.db2 = decoder_block(1024, 256)
        self.db3 = decoder_block(512, 128)
        self.db4 = decoder_block(256, 64)
        self.db5 = decoder_block(192, 32)

        self.classifier1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))

        self.classifier2 = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.interpo = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine = ref_seg()
        self._init_weight()

    def forward(self, input1, input2):
        input1_1, input2_1, input3_1, input4_1, input5_1 = input1[0], input1[1], input1[2], input1[3], input1[4]
        input1_2, input2_2, input3_2, input4_2, input5_2 = input2[0], input2[1], input2[2], input2[3], input2[4]

        x = torch.cat((input5_1, input5_2), dim=1)
        x = self.db1(x)

        # 512*16*16
        al1, x = self.db2(input4_1, x, input4_2)  # 256*32*32
        al2, x = self.db3(input3_1, x, input3_2)  # 128*64*64
        al3, x = self.db4(input2_1, x, input2_2)  # 64*128*128
        al4, x = self.db5(input1_1, x, input1_2)  # 32*256*256

        edge = self.classifier1(x)
        seg = self.classifier2(torch.cat((x, self.interpo(al4)), 1))
        result = self.refine(x, seg, edge)

        return al1, al2, al3, al4, result, seg

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
