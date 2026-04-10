import torch
import torch.nn as nn

from .lora_linear import CLoRALinear
from src.libs.logger.log import getLogger


class CLoRAAttention(nn.Module):
    def __init__(self, attn_layer: nn.MultiheadAttention, config):
        """
        Wraps an attentionLayer and injects LoRA into its key linear layers.
        """
        super().__init__()
        self._logger = getLogger(self.__class__.__name__)
        self.attn_layer = attn_layer

        # LoRA on the output projection of MultiheadAttention
        if "o_proj" in config.target_modules:
            self.attn_layer.self_attn.out_proj = CLoRALinear(
                self.attn_layer.self_attn.out_proj, config)

        # LoRA on the feed-forward layers
        if "fc1" in config.target_modules:
            self.attn_layer.linear1 = CLoRALinear(
                self.attn_layer.linear1, config)
        if "fc2" in config.target_modules:
            self.attn_layer.linear2 = CLoRALinear(
                self.attn_layer.linear2, config)

    def forward(self, *args, **kwargs):
        return self.attn_layer(*args, **kwargs)

