import torch
import torch.nn as nn
import torch.nn.functional as F
from scan import scan

from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding