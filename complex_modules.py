import math
import torch
import torch.nn as nn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def silu(x):
    return x * torch.sigmoid(x)


class ComplexPosition(nn.Module):
    """xpos used in RetNet"""

    def __init__(self, hidden_size, num_heads, theta=10000, complex_dtype=torch.complex64):
        super().__init__()
        self.complex_dtype = complex_dtype
        # taken from RoPE
        # self.theta = 1. / (theta**(torch.arange(0, hidden_size * 2, 2)[:hidden_size].float() /
        #                            (hidden_size * 2)))
        # self.theta = nn.Parameter(self.theta.view(num_heads, -1), requires_grad=False)
        self.theta = torch.randn(hidden_size) / hidden_size
        self.theta = nn.Parameter(self.theta.view(num_heads, -1))
        self.i = nn.Parameter(torch.complex(torch.tensor(0.0), torch.tensor(1.0)),
                              requires_grad=False)

    def forward(self, q, k, sequence_dim=-2, offset=0):
        """
        q,  # bsz * num_head * len * qk_dim
        k,  # bsz * num_head * len * qk_dim
        """
        if q.dtype != self.complex_dtype:
            q = q.to(self.complex_dtype)
            k = k.to(self.complex_dtype)

        seqlen = q.shape[sequence_dim]
        Theta = []

        for n in range(1, seqlen + 1):
            Theta.append(torch.exp(self.i * (n + offset) * self.theta))

        Theta = torch.stack(Theta, dim=1)  # [h, seqlen, d_k]
        Theta_bar = Theta.conj()

        Q = q * Theta.unsqueeze(0)
        K = k * Theta_bar.unsqueeze(0)

        return Q, K


class ComplexGroupNorm(nn.Module):

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(ComplexGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(num_channels, dtype=torch.float32))

    def forward(self, x):
        """
        x: (N, C, *)
        x is assumed to be complex
        """
        x = x.transpose(0, 1)  # (C, N, *)
        x_shape = x.shape
        x = x.reshape(self.num_groups, self.num_channels // self.num_groups, -1)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.reshape(self.num_channels, -1)
        if self.affine:
            x = x * self.weight + self.bias
        x = x.reshape(x_shape).transpose(0, 1)
        return x


class ComplexLayerNorm(nn.Module):

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(ComplexLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(num_channels, dtype=torch.float32))

    def forward(self, x):
        """
        x: unknown shape ending in hidden_size
        we treat the last dimension as the hidden_size
        """
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        mean = x.mean(dim=1, keepdim=True)
        var = x.abs().var(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        x = x.reshape(x_shape)
        return x


class ComplexLinear(nn.Linear):

    def forward(self, x):
        """
        x is assumed to be complex
        """
        # reshaping
        x = x @ self.weight.T.to(x)
        if self.bias is not None:
            x = x + self.bias.to(x)
        return x


class ComplexFFN(nn.Module):
    """
    2 linear layers + gelu
    """

    def __init__(self, hidden_size, mlp_proj_size, use_bias=False):
        super().__init__()
        self.layer1 = ComplexLinear(hidden_size, mlp_proj_size, bias=use_bias)
        self.layer2 = ComplexLinear(mlp_proj_size, hidden_size, bias=use_bias)
        self.activation = gelu

    def forward(self, x):
        """
        x is assumed to be complex
        """
        return self.layer2(self.activation(self.layer1(x)))
