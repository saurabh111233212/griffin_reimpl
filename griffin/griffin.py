import torch
import torch.nn as nn
import torch.nn.functional as F
from .scan import scan

from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-8, bias=False):
        super().__init__()
        
        self.dim = dim
        self.epsilon = epsilon
        self.bias = bias
        
        self.weight = nn.Parameter(torch.ones(dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = 0
    
    def forward(self, x):
        # x.shape = (batch, dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        x_normed = x / rms
        self.weight * x_normed + self.bias
        
                

class GatedMLP(nn.Module):
    pass

class RecurrentBlock(nn.Module):
    pass

class ResidualBlock(nn.Module):
    pass

class Griffin(nn.Module):
    pass