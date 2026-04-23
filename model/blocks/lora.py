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


class DoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=4, alpha=16, dropout=0.05, eps=1e-6):
        super().__init__()
        assert isinstance(base_layer, nn.Linear), "DoRA can only be applied to nn.Linear layers."

        self.base_layer = base_layer
        self.r = int(r)
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = alpha / r if r > 0 else 0.0
        self.eps = float(eps)

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.lora_A = None
        self.lora_B = None
        self.dora_magnitude = None
        if self.r > 0:
            self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            with torch.no_grad():
                init_magnitude = base_layer.weight.detach().float().norm(dim=1)
            self.dora_magnitude = nn.Parameter(init_magnitude)

    def _reshape_scale(self, ref_tensor, scale):
        shape = [1] * (ref_tensor.dim() - 1) + [scale.numel()]
        return scale.view(*shape)

    def forward(self, x):
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return self.base_layer(x)

        base_result = self.base_layer(x)
        base_linear = nn.functional.linear(x, self.base_layer.weight, bias=None)
        lora_update = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

        delta_weight = torch.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
        adapted_weight = self.base_layer.weight + delta_weight.to(
            dtype=self.base_layer.weight.dtype
        )
        weight_norm = adapted_weight.float().norm(dim=1).clamp_min(self.eps).detach()
        magnitude_scale = self._reshape_scale(
            base_linear,
            self.dora_magnitude.to(weight_norm.dtype) / weight_norm,
        ).to(dtype=base_linear.dtype)

        return (
            base_result
            + (magnitude_scale - 1.0) * base_linear
            + magnitude_scale * lora_update
        )


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
        self.register_buffer(
            "rank_center", torch.tensor(float(self.r_max), dtype=torch.float32)
        )
        self.register_buffer("probe_score", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("probe_block_score", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "probe_module_residual", torch.tensor(0.0, dtype=torch.float32)
        )
        self.register_buffer("probe_rank_prior", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("probe_selected", torch.tensor(True, dtype=torch.bool))
        self.register_buffer(
            "counterfactual_confirm", torch.zeros(self.r_max, dtype=torch.int64)
        )

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
    def get_rank_mask(self):
        return self.rank_mask.clone()

    @torch.no_grad()
    def set_rank_center(self, rank_center: float):
        rank_center = float(max(0.0, min(float(rank_center), float(self.r_max))))
        self.rank_center.fill_(rank_center)

    @torch.no_grad()
    def set_probe_score(self, score: float):
        self.probe_score.fill_(float(score))

    @torch.no_grad()
    def set_probe_prior(
        self,
        block_score: float,
        module_residual: float,
        rank_prior: float,
    ):
        self.probe_block_score.fill_(float(block_score))
        self.probe_module_residual.fill_(float(module_residual))
        self.probe_rank_prior.fill_(float(rank_prior))

    @torch.no_grad()
    def set_probe_selected(self, selected: bool):
        self.probe_selected.fill_(bool(selected))

    @torch.no_grad()
    def update_counterfactual_confirm(self, candidate_mask, tested_mask, safe_mask):
        candidate_mask = candidate_mask.to(
            device=self.counterfactual_confirm.device, dtype=torch.bool
        ).view(-1)
        tested_mask = tested_mask.to(
            device=self.counterfactual_confirm.device, dtype=torch.bool
        ).view(-1)
        safe_mask = safe_mask.to(
            device=self.counterfactual_confirm.device, dtype=torch.bool
        ).view(-1)
        if (
            candidate_mask.numel() != self.r_max
            or tested_mask.numel() != self.r_max
            or safe_mask.numel() != self.r_max
        ):
            raise ValueError("Counterfactual masks must match the searchable LoRA rank.")

        # Non-candidates and untested candidates should not accumulate stale confirmations.
        self.counterfactual_confirm.masked_fill_(~candidate_mask, 0)
        self.counterfactual_confirm.masked_fill_(candidate_mask & ~tested_mask, 0)
        self.counterfactual_confirm.masked_fill_(tested_mask & ~safe_mask, 0)
        self.counterfactual_confirm[safe_mask & tested_mask] += 1

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


class SpectralSearchableLoRALinear(nn.Module):
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
        spectral_prior_power=0.5,
        spectral_uncertainty_weight=0.5,
        spectral_init_scale=0.0,
    ):
        super().__init__()
        assert isinstance(base_layer, nn.Linear), "Spectral LoRA can only be applied to nn.Linear layers."
        if r <= 0:
            raise ValueError("SpectralSearchableLoRALinear expects a positive max rank.")

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
        self.spectral_prior_power = float(spectral_prior_power)
        self.spectral_uncertainty_weight = float(spectral_uncertainty_weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = self.alpha_over_r

        for p in self.base_layer.parameters():
            p.requires_grad = False

        u, singular_values, vh = self._truncated_svd(
            base_layer.weight.detach(),
            self.r_max,
        )
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.register_buffer("spectral_u", u.to(device=device, dtype=dtype))
        self.register_buffer("spectral_vh", vh.to(device=device, dtype=dtype))

        prior = singular_values.float()
        prior = prior / prior.median().clamp_min(1e-6)
        self.register_buffer("spectral_prior", prior.to(device=device))
        self.spectral_scale = nn.Parameter(
            torch.full(
                (self.r_max,),
                float(spectral_init_scale),
                dtype=torch.float32,
                device=device,
            )
        )

        self.register_buffer("rank_mask", torch.ones(self.r_max, dtype=torch.float32))
        self.register_buffer("active_rank", torch.tensor(self.r_max, dtype=torch.int64))
        self.register_buffer("importance_ema", torch.zeros(self.r_max, dtype=torch.float32))
        self.register_buffer("scale_grad_ema", torch.zeros(self.r_max, dtype=torch.float32))
        self.register_buffer("uncertainty_ema", torch.zeros(self.r_max, dtype=torch.float32))
        self.register_buffer("importance_ema_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("scale_grad_ema_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("uncertainty_ema_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "rank_center", torch.tensor(float(self.r_max), dtype=torch.float32)
        )
        self.register_buffer("probe_score", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("probe_block_score", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "probe_module_residual", torch.tensor(0.0, dtype=torch.float32)
        )
        self.register_buffer("probe_rank_prior", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("probe_selected", torch.tensor(True, dtype=torch.bool))
        self.register_buffer(
            "counterfactual_confirm", torch.zeros(self.r_max, dtype=torch.int64)
        )

        self.spectral_scale.register_hook(self._scale_grad_hook)

    @staticmethod
    def _truncated_svd(weight: torch.Tensor, rank: int):
        weight_cpu = weight.float().cpu()
        max_rank = min(weight_cpu.shape)
        rank = int(max(1, min(rank, max_rank)))
        q = min(max_rank, rank + 8)
        try:
            u, singular_values, v = torch.svd_lowrank(weight_cpu, q=q, niter=4)
            u = u[:, :rank].contiguous()
            singular_values = singular_values[:rank].contiguous()
            vh = v[:, :rank].t().contiguous()
        except RuntimeError:
            u, singular_values, vh = torch.linalg.svd(
                weight_cpu,
                full_matrices=False,
            )
            u = u[:, :rank].contiguous()
            singular_values = singular_values[:rank].contiguous()
            vh = vh[:rank, :].contiguous()
        return u, singular_values, vh

    @staticmethod
    def _ema_update(buffer, values, ready_flag, decay):
        if bool(ready_flag.item()):
            buffer.mul_(decay).add_(values * (1.0 - decay))
        else:
            buffer.copy_(values)
            ready_flag.fill_(True)

    def _scale_grad_hook(self, grad):
        if grad is None:
            return grad
        with torch.no_grad():
            values = grad.detach().float().abs()
            self._ema_update(
                self.scale_grad_ema,
                values,
                self.scale_grad_ema_ready,
                self.score_ema_decay,
            )
        return grad

    def raw_importance_scores(self):
        scale_scores = self.spectral_scale.detach().float().abs()
        if bool(self.scale_grad_ema_ready.item()):
            grad_scores = self.scale_grad_ema.clamp_min(0.0)
            if self.grad_weight > 0 and bool(grad_scores.gt(0).any().item()):
                grad_scale = grad_scores / grad_scores.median().clamp_min(1e-6)
                scale_scores = scale_scores * grad_scale.clamp_min(1e-6).pow(
                    self.grad_weight
                )
        if bool(scale_scores.gt(0).any().item()):
            return scale_scores
        return self.spectral_prior.detach().float().clone()

    def importance_scores(self):
        raw_scores = self.raw_importance_scores()
        with torch.no_grad():
            previous_scores = self.importance_ema.clone()
            was_ready = bool(self.importance_ema_ready.item())
            self._ema_update(
                self.importance_ema,
                raw_scores,
                self.importance_ema_ready,
                self.score_ema_decay,
            )
            scores = self.importance_ema.clone()
            if was_ready:
                uncertainty = (raw_scores - previous_scores).abs()
                self._ema_update(
                    self.uncertainty_ema,
                    uncertainty,
                    self.uncertainty_ema_ready,
                    self.score_ema_decay,
                )
            if (
                self.spectral_uncertainty_weight > 0
                and bool(self.uncertainty_ema_ready.item())
                and bool(self.uncertainty_ema.gt(0).any().item())
            ):
                uncertainty_scale = self.uncertainty_ema / self.uncertainty_ema.median().clamp_min(1e-6)
                scores = scores * (1.0 + uncertainty_scale).pow(
                    self.spectral_uncertainty_weight
                )
            if self.spectral_prior_power > 0:
                scores = scores * self.spectral_prior.float().clamp_min(1e-6).pow(
                    self.spectral_prior_power
                )
        return scores

    def probe_gradient_score(self):
        grad = self.spectral_scale.grad
        if grad is None:
            return None
        return float(grad.detach().float().pow(2).mean().sqrt().item())

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
    def get_rank_mask(self):
        return self.rank_mask.clone()

    @torch.no_grad()
    def set_rank_center(self, rank_center: float):
        rank_center = float(max(0.0, min(float(rank_center), float(self.r_max))))
        self.rank_center.fill_(rank_center)

    @torch.no_grad()
    def set_probe_score(self, score: float):
        self.probe_score.fill_(float(score))

    @torch.no_grad()
    def set_probe_prior(
        self,
        block_score: float,
        module_residual: float,
        rank_prior: float,
    ):
        self.probe_block_score.fill_(float(block_score))
        self.probe_module_residual.fill_(float(module_residual))
        self.probe_rank_prior.fill_(float(rank_prior))

    @torch.no_grad()
    def set_probe_selected(self, selected: bool):
        self.probe_selected.fill_(bool(selected))

    @torch.no_grad()
    def update_counterfactual_confirm(self, candidate_mask, tested_mask, safe_mask):
        candidate_mask = candidate_mask.to(
            device=self.counterfactual_confirm.device, dtype=torch.bool
        ).view(-1)
        tested_mask = tested_mask.to(
            device=self.counterfactual_confirm.device, dtype=torch.bool
        ).view(-1)
        safe_mask = safe_mask.to(
            device=self.counterfactual_confirm.device, dtype=torch.bool
        ).view(-1)
        if (
            candidate_mask.numel() != self.r_max
            or tested_mask.numel() != self.r_max
            or safe_mask.numel() != self.r_max
        ):
            raise ValueError("Counterfactual masks must match the searchable LoRA rank.")

        self.counterfactual_confirm.masked_fill_(~candidate_mask, 0)
        self.counterfactual_confirm.masked_fill_(candidate_mask & ~tested_mask, 0)
        self.counterfactual_confirm.masked_fill_(tested_mask & ~safe_mask, 0)
        self.counterfactual_confirm[safe_mask & tested_mask] += 1

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

        projected = nn.functional.linear(
            self.dropout(x),
            self.spectral_vh.to(dtype=x.dtype),
        )
        scale = self.spectral_scale * self.rank_mask.to(
            device=self.spectral_scale.device,
            dtype=self.spectral_scale.dtype,
        )
        scale = scale.to(dtype=projected.dtype)
        scale_shape = [1] * (projected.dim() - 1) + [self.r_max]
        update = projected * scale.view(*scale_shape)
        update = nn.functional.linear(
            update,
            self.spectral_u.to(dtype=update.dtype),
        )
        return base + update * self.scaling
