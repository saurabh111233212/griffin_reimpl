from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding

from .scan import scan


@dataclass
class GriffinConfig:
    vocab_size: int = 1
    n_layers: int = 1
    dim: int = 256


class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-8, bias=False):
        super().__init__()

        self.dim = dim
        self.epsilon = epsilon
        self.has_bias = bias

        self.weight = nn.Parameter(torch.ones(dim))
        if self.has_bias:
            self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x.shape = (batch, dim)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        x_normed_scaled = self.weight * (x / rms)
        if self.has_bias:
            x_normed_scaled += self.b
        return x_normed_scaled

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.fill_(1)
            if self.bias:
                self.b.fill_(0)


class GatedMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()

        self.up_proj1 = nn.Linear(in_dim, hidden_dim)
        self.up_proj2 = nn.Linear(in_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, out_dim)
        self.gelu = nn.GELU()

        self.reset_parameters()

    def forward(self, x):
        left_side = self.gelu(self.up_proj1(x))
        right_side = self.up_proj2(x)
        return self.down_proj(left_side * right_side)

    def reset_parameters(self):
        with torch.no_grad():
            in_dim = self.up_proj1.in_features
            hidden_dim = self.up_proj1.out_features
            self.up_proj1.weight.normal_(std=in_dim**0.5)
            self.up_proj2.weight.normal_(std=in_dim**0.5)
            self.down_proj.weight.normal_(std=hidden_dim**0.5)


# THIS IS WHERE THE CUDA SCAN WILL BE USED
class TempConv1D(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def reset_parameters(self):
        pass


# the "real gated linear recurrent unit" (RGLRU)
class RGLRU(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(self, x):
        pass

    def reset_parameters(self):
        pass


class RecurrentBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim, hidden_dim):
        super().__init__()
        self.left_linear = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.right_linear = nn.Linear(in_dim, hidden_dim)
        self.temp_conv_1d = TempConv1D()
        self.rg_lru = RGLRU()

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        left_x = self.gelu(self.left_linear(x))
        right_x = self.rg_lru(self.temp_conv_1d(self.right_linear(x)))
        return self.out(left_x * right_x)

    def reset_parameters(self):
        self.left_linear.reset_parameters()
        self.right_linear.reset_parameters()
        self.temp_conv_1d.reset_parameters()
        self.rg_lru.reset_parameters()
        self.out.reset_parameters()


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = RMSNorm()
        self.recurrent = RecurrentBlock()
        self.norm2 = RMSNorm()
        self.gated_mlp = GatedMLP()

    def forward(self, x):
        x_1 = self.recurrent(self.norm1(x)) + x
        return self.gated_mlp(self.norm2(x_1)) + x_1

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.recurrent.reset_parameters()
        self.norm2.reset_parameters()
        self.gated_mlp.reset_parameters()


class Griffin(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def reset_parameters(self):
        pass
