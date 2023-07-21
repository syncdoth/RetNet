import torch


def split_chunks(*tensors, size, dim=0):
    return (torch.split(x, size, dim=dim) for x in tensors)


def split_heads(tensors, bsz, seqlen, num_heads):
    assert isinstance(tensors, (tuple, list))
    return (x.view(bsz, seqlen, num_heads, -1).transpose(1, 2) for x in tensors)
