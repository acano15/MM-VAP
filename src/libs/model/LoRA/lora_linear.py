import math
import torch
import torch.nn as nn

from src.libs.logger.log import getLogger


class CLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: dict = None):
        super().__init__()
        self.config = config

        # Keep original Linear
        self.base_layer = base_layer

        # Expose weight/bias for compatibility
        self.weight = self.base_layer.weight
        self.bias = self.base_layer.bias

        # LoRA parameters
        r = config.rank
        alpha = config.alpha
        dropout = config.dropout
        self.scaling = alpha / r

        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # init LoRA params
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.dropout(self.lora_B(self.lora_A(x))) * self.scaling
        return base_out + lora_out
