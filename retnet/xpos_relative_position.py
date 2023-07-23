# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/microsoft/torchscale/blob/2b101355d79dc48b8cdf4bc58a94f98be69f182a/torchscale/component/xpos_relative_position.py

import torch
import torch.nn as nn


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000**(torch.arange(0, dim) / dim))
    sinusoid_inp = (torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float),
                                 inv_freq).to(x))
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):

    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer("scale",
                             (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim))

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        # min_pos = -(length + offset) // 2
        min_pos = 0  # for recurrence, no negative min_pos
        max_pos = length + offset + min_pos
        scale = self.scale**torch.arange(min_pos, max_pos,
                                         1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

    def rotate_queries_and_keys(self, q, k, offset=0):
        q = self.forward(q, offset=offset, downscale=False)
        k = self.forward(k, offset=offset, downscale=True)
        return q, k


if __name__ == "__main__":
    # test
    seqlen = 8
    q = torch.randn(1, seqlen, 4)  # [b, t, d]
    k = torch.randn(1, seqlen, 4)  # [b, t, d]
    xpos = XPOS(4, scale_base=512)
    # parallel
    p_q_r, p_k_r = xpos.rotate_queries_and_keys(q, k)
    # recurrent
    q_r, k_r = [], []
    for i in range(seqlen):
        _q, _k = xpos.rotate_queries_and_keys(q[:, i:i + 1], k[:, i:i + 1], offset=i)
        q_r.append(_q)
        k_r.append(_k)
    q_r = torch.cat(q_r, dim=1)
    k_r = torch.cat(k_r, dim=1)

    # check
    print('query')
    print('parallel')
    print(p_q_r)
    print('recurrent')
    print(q_r)
    print('key')
    print('parallel')
    print(p_k_r)
    print('recurrent')
    print(k_r)
