import torch.nn as nn
from models import register

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
@register('fusionnet-resnet')
class FusionNet(nn.Module):

    def __init__(self, in_dim, out_dim, n_resblocks, conv=default_conv):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_resblocks = n_resblocks
        net = [
            ResBlock(
                conv, self.in_dim
            ) for _ in range(n_resblocks)
        ]
        net.append(conv(self.in_dim, self.out_dim, 3))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
