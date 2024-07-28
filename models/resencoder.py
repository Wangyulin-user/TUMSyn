# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.cross_att import LayerNorm


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=True, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm3d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResEncoder(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(ResEncoder, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        out_dim = args.out_dim
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = out_dim
            m_tail = [conv(n_feats, self.out_dim, kernel_size)]
            self.tail = nn.Sequential(*m_tail)
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class ResDecoder(nn.Module):
    def __init__(self):
        super(ResDecoder, self).__init__()
        self.conv = default_conv
        n_resblocks = 3
        out_dim = 768
        kernel_size = 3
        act = nn.ReLU(True)
        # define body module
        m_body = [
            ResBlock(
                self.conv, out_dim, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(self.conv(out_dim, out_dim, kernel_size))

        self.body = nn.Sequential(*m_body)


        self.out_dim = out_dim
        m_tail = [self.conv(out_dim, out_dim, kernel_size)]
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('resencoder-64')
def make_edsr_baseline(n_resblocks=16, n_feats=64, outdim=128,
                       res_scale=1, scale=1, no_upsampling=True):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.out_dim = outdim
    args.res_scale = res_scale
    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 1
    return ResEncoder(args)


@register('resencoder-256')
def make_edsr(n_resblocks=24, n_feats=256, outdim=768,
              res_scale=0.1, scale=1, no_upsampling=True):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.out_dim = outdim
    args.res_scale = res_scale
    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 1
    return ResEncoder(args)


