import torch
from torch import nn
from net_v4.models.vmamba2 import VSSM, LayerNorm2d

#进入编码层第一件干什么事
#参照MambaCD
#应该是stem
#但是他用的VSSM的Backbone，我是也用这个然后在这上面修改还是自己重新写？
#修改的话，改内部估计改不动，所以还是重新写
#先把Vmamba爬过来

class encoder(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d',**kwargs):
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        kwargs.update(norm_layer=norm_layer,channel_first=self.channel_first)
        super().__init__(**kwargs)

        #也就是每个输出有自己独立的norm
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
            #会打印出匹配的预训练权重和未匹配的权重
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, A, B):
        #一个block封装了好几个VSSblock和下采样
        #一个block对应一个layer，一个layer对应多个VSSblock和一个下采样
        #所以要把VSSblock替换掉，只用把这里修改成把A,B都传进去
        def layer_forward(l, A, B):
            b = l.blocks([A,B])
            #forward只能有一个数据传入,所以合并拆分
            #通过sequential封装类，forward固定传入一个数据
            #通过
            b1 = b[0]
            b2 = b[1]
            #下采样共享参数
            d = l.ddb(b)
            bd1 = l.downsample(b1)
            bd2 = l.downsample(b2)
            return b1,b2, d, bd1,bd2
        bd1 = self.patch_embed(A)
        bd2 = self.patch_embed(B)
        outs = []
        for i, layer in enumerate(self.layers):
            b1,b2, d, bd1,bd2 = layer_forward(layer, bd1, bd2)  # (B, H, W, C)
            #o是经过block的输出，x是经过了block和下采样的输出,下采样继续往后走
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out1 = norm_layer(b1)
                out2 = norm_layer(b2)
                #是不是d应该也加一个归一呢,激活前也有归一应该不用了
                if not self.channel_first:
                    #使张量连续存储
                    out1 = out1.permute(0, 3, 1, 2).contiguous()
                    out2 = out2.permute(0, 3, 1, 2).contiguous()
                    d = d.permute(0, 3, 1, 2).contiguous()
                outs.append(out1)
                outs.append(out2)
                outs.append(d)

        #怎么可能为0？逻辑应该是判断走到走后一层了，然后输出bd
        if len(self.out_indices) == 0:
            return bd1,bd2
        return outs


