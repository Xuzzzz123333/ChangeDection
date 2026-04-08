import math
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=4, alpha=16, dropout=0.05):
        super().__init__()
        assert isinstance(base_layer, nn.Linear), "LoRA can only be applied to nn.Linear layers."

        self.base_layer = base_layer
        self.r = r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = alpha / r if r > 0 else 1.0

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
