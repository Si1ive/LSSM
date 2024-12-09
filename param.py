import torch

from fvcore.nn import FlopCountAnalysis ,parameter_count_table
from net_v3.ZZHnet import zzh_net
from thop import profile, clever_format

if __name__ == "__main__":
    model = zzh_net(num_class=2)
    model.cuda().train()

    A = torch.randn((1, 3, 256, 256)).cuda()
    B = torch.randn((1, 3, 256, 256)).cuda()

    macs, params = profile(model, inputs=(A,B,))
    print('thop'+'-' * 60)
    print("Params:%.2fM | macs:%.2fG" % ( params / (1000 ** 2), macs / (1000 ** 3)))

    flops = FlopCountAnalysis(model, (A,B))
    print('fvcore' + '-' * 60)
    print('FLOPS:{:.2f}G'.format(flops.total() / (1000 ** 3)))
    print(parameter_count_table(model))


