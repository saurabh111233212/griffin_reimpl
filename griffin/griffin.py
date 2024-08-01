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


# the "real gated linear recurrent unit" (RGLRU)
class RGLRU(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.recurrence_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.c = 8.0
        self.lambd = nn.Parameter(torch.empty(self.hidden_dim))

        self.reset_parameters()

    def forward(self, x):
        r = nn.Sigmoid(self.recurrence_gate(x))  # batch, seq_len, h_dim
        i = nn.Sigmoid(self.input_gate(x))  # batch, seq_len, h_dim
        a = torch.pow(nn.Sigmoid(self.lambd), self.c * r)  # batch, seq_len, h_dim
        # h = a * h_prev + torch.sqrt(1 - a**2) * (i * x) #scan here?? OHHHHH it's a scan sweeping accross t
        h = scan()
        # ^  batch, seq_len, h_dim

        return h

    def reset_parameters(self):
        pass


class RecurrentBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.left_linear = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.right_linear = nn.Linear(in_dim, hidden_dim)
        self.temp_conv_1d = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=4, groups=hidden_dim, padding=3
        )
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


class SlidingGQA(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_dim: int,
        q_heads: int,
        kv_heads: int,
        window_size: int,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.pe = RotaryEmbedding(dim=head_dim)
        self.q_proj = nn.Linear(hidden_dim, head_dim * q_heads, bias=False)
        self.kv_proj = nn.Linear(hidden_dim, 2 * head_dim * kv_heads, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, -1, self.hidden_dim
        )  # batch, seq_len, q_heads, hidden_dim
        kv = self.kv_proj(x).view(
            batch_size, seq_len, 2, -1, self.hidden_dim
        )  # batch, seq_len, 2 (one for k and one for v), kv_heads, hidden_dim
        q, kv = self.pe(q, kv)
        x = flash_attn_func(
            q, kv[:, :, 0], causal=True, window_size=(-self.window_size, 0)
        )
        x = x.view(batch_size, seq_len, hidden_dim)

        return self.out(x)

    def reset_parameters(self):
        self.q_proj.weight.normal_(std=self.hidden_dim**-0.5)
        self.kv_proj.weight.normal_(std=self.hidden_dim**-0.5)
        self.out.weight.normal_(std=self.hidden_dim**-0.5)


class Griffin(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def reset_parameters(self):
        pass
