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
    def __init__(
        self,
        base_layer: nn.Linear,
        r=8,
        alpha_over_r=1.0,
        dropout=0.05,
        module_name="",
        group_name="other",
        score_ema_decay=0.9,
        grad_weight=0.5,
    ):
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
        self.module_name = module_name
        self.group_name = group_name
        self.score_ema_decay = float(score_ema_decay)
        self.grad_weight = float(grad_weight)
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
        self.register_buffer("importance_ema", torch.zeros(self.r_max, dtype=torch.float32))
        self.register_buffer("grad_a_ema", torch.zeros(self.r_max, dtype=torch.float32))
        self.register_buffer("grad_b_ema", torch.zeros(self.r_max, dtype=torch.float32))
        self.register_buffer("importance_ema_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("grad_a_ema_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("grad_b_ema_ready", torch.tensor(False, dtype=torch.bool))

        self.lora_A.weight.register_hook(self._make_grad_hook("a"))
        self.lora_B.weight.register_hook(self._make_grad_hook("b"))

    def raw_importance_scores(self):
        a_norm = self.lora_A.weight.float().pow(2).sum(dim=1).sqrt()
        b_norm = self.lora_B.weight.float().pow(2).sum(dim=0).sqrt()
        return a_norm * b_norm

    @staticmethod
    def _ema_update(buffer, values, ready_flag, decay):
        if bool(ready_flag.item()):
            buffer.mul_(decay).add_(values * (1.0 - decay))
        else:
            buffer.copy_(values)
            ready_flag.fill_(True)

    def _make_grad_hook(self, branch: str):
        def hook(grad):
            if grad is None:
                return grad
            with torch.no_grad():
                if branch == "a":
                    values = grad.float().pow(2).sum(dim=1).sqrt()
                    self._ema_update(
                        self.grad_a_ema,
                        values,
                        self.grad_a_ema_ready,
                        self.score_ema_decay,
                    )
                else:
                    values = grad.float().pow(2).sum(dim=0).sqrt()
                    self._ema_update(
                        self.grad_b_ema,
                        values,
                        self.grad_b_ema_ready,
                        self.score_ema_decay,
                    )
            return grad

        return hook

    def importance_scores(self):
        raw_scores = self.raw_importance_scores()
        with torch.no_grad():
            self._ema_update(
                self.importance_ema,
                raw_scores,
                self.importance_ema_ready,
                self.score_ema_decay,
            )
            scores = self.importance_ema.clone()
            if (
                self.grad_weight > 0
                and bool(self.grad_a_ema_ready.item())
                and bool(self.grad_b_ema_ready.item())
            ):
                grad_scores = torch.sqrt(
                    self.grad_a_ema.clamp_min(0.0) * self.grad_b_ema.clamp_min(0.0)
                )
                if bool(grad_scores.gt(0).any().item()):
                    grad_scale = grad_scores / grad_scores.median().clamp_min(1e-6)
                    scores = scores * grad_scale.clamp_min(1e-6).pow(self.grad_weight)
        return scores

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

        scores = self.raw_importance_scores()
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
