import math

import torch
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
        self.scaling = alpha / r if r > 0 else 0.0

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.lora_A = None
        self.lora_B = None
        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False)

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return self.base_layer(x)
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class SearchableLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, alpha_over_r=1.0, dropout=0.05):
        super().__init__()
        assert isinstance(base_layer, nn.Linear), "LoRA can only be applied to nn.Linear layers."
        if r <= 0:
            raise ValueError("SearchableLoRALinear expects a positive max rank.")

        self.base_layer = base_layer
        self.r_max = int(r)
        self.r = self.r_max
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.alpha_over_r = float(alpha_over_r)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = self.alpha_over_r

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(self.in_features, self.r_max, bias=False)
        self.lora_B = nn.Linear(self.r_max, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.register_buffer("rank_mask", torch.ones(self.r_max, dtype=torch.float32))
        self.register_buffer("active_rank", torch.tensor(self.r_max, dtype=torch.int64))

    def importance_scores(self):
        a_norm = self.lora_A.weight.float().pow(2).sum(dim=1).sqrt()
        b_norm = self.lora_B.weight.float().pow(2).sum(dim=0).sqrt()
        return a_norm * b_norm

    @torch.no_grad()
    def set_rank_mask(self, mask):
        mask = mask.to(device=self.rank_mask.device, dtype=self.rank_mask.dtype).view(-1)
        if mask.numel() != self.r_max:
            raise ValueError(
                f"Rank mask size mismatch: expected {self.r_max}, got {mask.numel()}."
            )
        self.rank_mask.copy_(mask)
        self.active_rank.fill_(int(mask.gt(0).sum().item()))

    @torch.no_grad()
    def set_active_rank(self, rank: int):
        rank = int(max(0, min(rank, self.r_max)))
        if rank == 0:
            self.set_rank_mask(torch.zeros_like(self.rank_mask))
            return
        if rank == self.r_max:
            self.set_rank_mask(torch.ones_like(self.rank_mask))
            return

        scores = self.importance_scores()
        keep_idx = scores.topk(rank, largest=True, sorted=False).indices
        mask = torch.zeros_like(scores)
        mask[keep_idx] = 1.0
        self.set_rank_mask(mask)

    def forward(self, x):
        base = self.base_layer(x)
        if int(self.active_rank.item()) <= 0:
            return base

        update = self.lora_A(self.dropout(x))
        update = update * self.rank_mask.to(dtype=update.dtype)
        update = self.lora_B(update) * self.scaling
        return base + update
