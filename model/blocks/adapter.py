import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import DoRALinear, LoRALinear, SearchableLoRALinear

REPO_DIR = "dinov3"
DINO_NAME = "dinov3_vitl16"
MODEL_TO_NUM_LAYERS = {
    "VITS": 12,
    "VITSP": 12,
    "VITB": 12,
    "VITL": 24,
    "VITHP": 32,
    "VIT7B": 40,
}


class DINOConvTokenBranch(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("DINOConvTokenBranch expects a positive odd kernel size.")
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=False,
        )
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    @staticmethod
    def infer_spatial_shape(num_patches: int):
        if num_patches <= 0:
            return 0, 0
        height = int(math.sqrt(num_patches))
        while height > 1 and num_patches % height != 0:
            height -= 1
        width = num_patches // height
        return height, width

    def forward(self, patch_tokens: torch.Tensor):
        batch_size, num_patches, channels = patch_tokens.shape
        height, width = self.infer_spatial_shape(num_patches)
        if height * width != num_patches:
            raise ValueError(
                f"Cannot infer a valid spatial shape from {num_patches} patch tokens."
            )
        feat = patch_tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        feat = self.depthwise(feat)
        feat = self.act(feat)
        feat = self.pointwise(feat)
        return feat.flatten(2).transpose(1, 2)


class DINOBlockLocalConvAdapter(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        dim: int,
        num_prefix_tokens: int,
        kernel_size: int = 3,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.block = block
        self.num_prefix_tokens = int(max(0, num_prefix_tokens))
        self.norm = nn.LayerNorm(dim)
        self.local_branch = DINOConvTokenBranch(dim=dim, kernel_size=kernel_size)
        self.gamma = nn.Parameter(torch.full((dim,), float(init_scale)))

    def _forward_tensor(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[1] <= self.num_prefix_tokens:
            return x

        normed = self.norm(x)
        patch_tokens = normed[:, self.num_prefix_tokens :]
        local_tokens = self.local_branch(patch_tokens)
        if self.num_prefix_tokens > 0:
            prefix_tokens = x[:, : self.num_prefix_tokens]
            patch_tokens = x[:, self.num_prefix_tokens :] + local_tokens * self.gamma.view(
                1, 1, -1
            )
            return torch.cat([prefix_tokens, patch_tokens], dim=1)
        return x + local_tokens * self.gamma.view(1, 1, -1)

    def forward(self, x_or_x_list, rope_or_rope_list=None):
        out = self.block(x_or_x_list, rope_or_rope_list)
        if isinstance(out, torch.Tensor):
            return self._forward_tensor(out)
        if isinstance(out, list):
            return [self._forward_tensor(x) for x in out]
        raise TypeError(f"Unsupported DINO block output type: {type(out)!r}")


class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        extract_ids=[5, 11, 17, 23],
        device="cuda",
        use_lora=False,
        use_dora=False,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_search=False,
        lora_r_target=None,
        lora_alpha_over_r=1.0,
        lora_search_warmup_epochs=5,
        lora_search_interval=1,
        lora_search_ema_decay=0.9,
        lora_search_score_norm="median",
        lora_search_grad_weight=0.5,
        lora_search_budget_mode="grouped",
        lora_search_group_weights=None,
        lora_search_depth_buckets=3,
        lora_search_counterfactual=False,
        lora_search_counterfactual_val_batches=2,
        lora_search_counterfactual_max_candidates=64,
        lora_search_counterfactual_delta=0.0,
        lora_search_counterfactual_patience=1,
        local_conv_enable=False,
        local_conv_blocks=(5, 11, 17, 23),
        local_conv_kernel_size=3,
        local_conv_init_scale=0.0,
    ):
        super().__init__()
        self.device = device
        self.use_dora = bool(use_dora)
        self.use_lora = bool(use_lora or self.use_dora)
        self.lora_search = self.use_lora and lora_search
        self.lora_r = int(lora_r)
        self.lora_r_target = (
            self.lora_r if lora_r_target is None else int(max(0, lora_r_target))
        )
        self.lora_alpha_over_r = float(lora_alpha_over_r)
        self.lora_search_warmup_epochs = int(max(0, lora_search_warmup_epochs))
        self.lora_search_interval = int(max(1, lora_search_interval))
        self.lora_search_ema_decay = float(lora_search_ema_decay)
        self.lora_search_score_norm = lora_search_score_norm
        self.lora_search_grad_weight = float(lora_search_grad_weight)
        self.lora_search_budget_mode = lora_search_budget_mode
        self.lora_search_depth_buckets = int(max(1, lora_search_depth_buckets))
        self.lora_search_counterfactual = bool(lora_search_counterfactual)
        self.lora_search_counterfactual_val_batches = int(
            max(0, lora_search_counterfactual_val_batches)
        )
        self.lora_search_counterfactual_max_candidates = int(
            max(1, lora_search_counterfactual_max_candidates)
        )
        self.lora_search_counterfactual_delta = float(
            lora_search_counterfactual_delta
        )
        self.lora_search_counterfactual_patience = int(
            max(1, lora_search_counterfactual_patience)
        )
        self.local_conv_enable = bool(local_conv_enable)
        self.local_conv_blocks = tuple(sorted(set(int(index) for index in local_conv_blocks)))
        self.local_conv_kernel_size = int(local_conv_kernel_size)
        self.local_conv_init_scale = float(local_conv_init_scale)
        self.lora_search_group_weights = {
            "attn.qkv": 1.0,
            "attn.proj": 1.0,
            "mlp.fc1": 1.0,
            "mlp.fc2": 1.0,
        }
        if lora_search_group_weights is not None:
            self.lora_search_group_weights.update(lora_search_group_weights)
        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            weights=weights_path,
        )
        self.model = self.model.eval().to(device)
        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = extract_ids

        for p in self.model.parameters():
            p.requires_grad = False

        if self.local_conv_enable:
            self.inject_local_conv(
                block_indices=self.local_conv_blocks,
                kernel_size=self.local_conv_kernel_size,
                init_scale=self.local_conv_init_scale,
            )

        if self.use_lora:
            self.inject_lora(
                self.model,
                r=self.lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                search=self.lora_search,
                use_dora=self.use_dora,
                alpha_over_r=self.lora_alpha_over_r,
                score_ema_decay=self.lora_search_ema_decay,
                grad_weight=self.lora_search_grad_weight,
                num_layers=self.n_layers,
                depth_buckets=self.lora_search_depth_buckets,
            )
            if self.lora_search:
                self.update_lora_rank_budget(self.lora_r)

    @staticmethod
    def should_apply_lora(full_name: str) -> bool:
        targets = (
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
        )
        return any(full_name.endswith(target) for target in targets)

    @staticmethod
    def _extract_layer_index(full_name: str):
        match = re.search(r"(?:^|\.)blocks\.(\d+)\.", full_name)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _get_depth_bucket_label(layer_index: int, num_layers: int, num_buckets: int) -> str:
        if num_buckets <= 1 or num_layers <= 1:
            return "all"
        bucket_index = min(
            num_buckets - 1,
            int(layer_index * num_buckets / max(1, num_layers)),
        )
        if num_buckets == 2:
            labels = ("early", "late")
            return labels[bucket_index]
        if num_buckets == 3:
            labels = ("early", "middle", "late")
            return labels[bucket_index]
        return f"depth{bucket_index}"

    @staticmethod
    def get_lora_group_name(full_name: str, num_layers: int, depth_buckets: int) -> str:
        targets = (
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
        )
        for target in targets:
            if full_name.endswith(target):
                if depth_buckets <= 1:
                    return target
                layer_index = DINOV3Wrapper._extract_layer_index(full_name)
                if layer_index is None:
                    return target
                bucket_label = DINOV3Wrapper._get_depth_bucket_label(
                    layer_index,
                    num_layers,
                    depth_buckets,
                )
                return f"{target}.{bucket_label}"
        return "other"

    @staticmethod
    def inject_lora(
        module,
        prefix="",
        r=4,
        alpha=16,
        dropout=0.05,
        search=False,
        use_dora=False,
        alpha_over_r=1.0,
        score_ema_decay=0.9,
        grad_weight=0.5,
        num_layers=24,
        depth_buckets=1,
    ):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and DINOV3Wrapper.should_apply_lora(full_name):
                if search:
                    if use_dora:
                        raise ValueError("Searchable DoRA is not supported yet.")
                    setattr(
                        module,
                        name,
                        SearchableLoRALinear(
                            child,
                            r=r,
                            alpha_over_r=alpha_over_r,
                            dropout=dropout,
                            module_name=full_name,
                            group_name=DINOV3Wrapper.get_lora_group_name(
                                full_name,
                                num_layers,
                                depth_buckets,
                            ),
                            score_ema_decay=score_ema_decay,
                            grad_weight=grad_weight,
                        ),
                    )
                else:
                    if use_dora:
                        setattr(
                            module,
                            name,
                            DoRALinear(child, r=r, alpha=alpha, dropout=dropout),
                        )
                    else:
                        setattr(
                            module,
                            name,
                            LoRALinear(child, r=r, alpha=alpha, dropout=dropout),
                        )
            else:
                DINOV3Wrapper.inject_lora(
                    child,
                    full_name,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    search=search,
                    use_dora=use_dora,
                    alpha_over_r=alpha_over_r,
                    score_ema_decay=score_ema_decay,
                    grad_weight=grad_weight,
                    num_layers=num_layers,
                    depth_buckets=depth_buckets,
                )

    def inject_local_conv(self, block_indices, kernel_size=3, init_scale=0.0):
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise ValueError("The loaded DINO backbone does not expose a blocks module list.")

        dim = int(getattr(self.model, "embed_dim"))
        num_prefix_tokens = 1 + int(getattr(self.model, "n_storage_tokens", 0))
        total_blocks = len(blocks)
        valid_indices = []
        for block_index in block_indices:
            if block_index < 0 or block_index >= total_blocks:
                raise ValueError(
                    f"Requested DINO local conv block index {block_index}, but the model only has {total_blocks} blocks."
                )
            valid_indices.append(int(block_index))

        for block_index in valid_indices:
            block = blocks[block_index]
            if isinstance(block, DINOBlockLocalConvAdapter):
                continue
            blocks[block_index] = DINOBlockLocalConvAdapter(
                block=block,
                dim=dim,
                num_prefix_tokens=num_prefix_tokens,
                kernel_size=kernel_size,
                init_scale=init_scale,
            ).to(self.device)

    def _resolve_group_weight(self, group_name: str) -> float:
        if group_name in self.lora_search_group_weights:
            return max(0.0, float(self.lora_search_group_weights[group_name]))
        for base_group_name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            if group_name == base_group_name or group_name.startswith(base_group_name + "."):
                return max(
                    0.0,
                    float(self.lora_search_group_weights.get(base_group_name, 1.0)),
                )
        return 1.0

    def iter_searchable_lora_layers(self):
        for module in self.model.modules():
            if isinstance(module, SearchableLoRALinear):
                yield module

    @staticmethod
    def _normalize_group_scores(score_chunks, mode: str):
        if not score_chunks:
            return score_chunks
        if mode == "none":
            return score_chunks

        flat_scores = torch.cat(score_chunks, dim=0)
        if mode == "median":
            scale = flat_scores.median().clamp_min(1e-6)
            return [scores / scale for scores in score_chunks]

        mean = flat_scores.mean()
        std = flat_scores.std(unbiased=False).clamp_min(1e-6)
        return [(scores - mean) / std for scores in score_chunks]

    def _allocate_group_budgets(self, total_budget: int, group_capacity: dict):
        if total_budget <= 0:
            return {group_name: 0 for group_name in group_capacity}

        weighted_capacity = {}
        for group_name, capacity in group_capacity.items():
            weight = self._resolve_group_weight(group_name)
            weighted_capacity[group_name] = capacity * weight

        if sum(weighted_capacity.values()) <= 0:
            weighted_capacity = {group_name: float(capacity) for group_name, capacity in group_capacity.items()}

        raw_budget = {
            group_name: total_budget * weighted_capacity[group_name] / sum(weighted_capacity.values())
            for group_name in group_capacity
        }
        allocated = {
            group_name: min(group_capacity[group_name], int(raw_budget[group_name]))
            for group_name in group_capacity
        }
        remaining = total_budget - sum(allocated.values())

        while remaining > 0:
            candidates = [
                group_name
                for group_name, capacity in group_capacity.items()
                if allocated[group_name] < capacity
            ]
            if not candidates:
                break
            candidates.sort(
                key=lambda group_name: (
                    raw_budget[group_name] - int(raw_budget[group_name]),
                    self._resolve_group_weight(group_name),
                    group_capacity[group_name] - allocated[group_name],
                ),
                reverse=True,
            )
            allocated[candidates[0]] += 1
            remaining -= 1

        return allocated

    def _compute_rank_budget(self, layers, budget_rank: int):
        budget_rank = int(max(0, min(budget_rank, self.lora_r)))
        total_capacity = sum(layer.r_max for layer in layers)
        total_budget = min(total_capacity, budget_rank * len(layers))
        return budget_rank, total_capacity, total_budget

    def _compute_normalized_scores(self, layers):
        layer_scores = [layer.importance_scores() for layer in layers]
        group_to_indices = {}
        group_capacity = {}
        for index, layer in enumerate(layers):
            group_name = layer.group_name
            group_to_indices.setdefault(group_name, []).append(index)
            group_capacity[group_name] = group_capacity.get(group_name, 0) + layer.r_max

        normalized_scores = [None] * len(layers)
        for group_name, indices in group_to_indices.items():
            group_scores = [layer_scores[index] for index in indices]
            group_scores = self._normalize_group_scores(
                group_scores,
                self.lora_search_score_norm,
            )
            for index, scores in zip(indices, group_scores):
                normalized_scores[index] = scores
        return normalized_scores, group_to_indices, group_capacity

    def _build_keep_masks(self, layers, normalized_scores, total_budget, total_capacity, group_to_indices, group_capacity):
        if total_budget <= 0:
            return [torch.zeros_like(scores) for scores in normalized_scores], {}
        if total_budget >= total_capacity:
            return [torch.ones_like(scores) for scores in normalized_scores], {}

        keep_masks = [torch.zeros_like(scores) for scores in normalized_scores]
        if self.lora_search_budget_mode == "global":
            flat_scores = torch.cat(normalized_scores, dim=0)
            keep_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            keep_idx = flat_scores.topk(total_budget, largest=True, sorted=False).indices
            keep_mask[keep_idx] = True

            offset = 0
            for index, layer in enumerate(layers):
                next_offset = offset + layer.r_max
                keep_masks[index] = keep_mask[offset:next_offset].to(
                    dtype=layers[index].rank_mask.dtype
                )
                offset = next_offset
            return keep_masks, {}

        group_budgets = self._allocate_group_budgets(total_budget, group_capacity)
        for group_name, indices in group_to_indices.items():
            group_budget = min(group_capacity[group_name], group_budgets.get(group_name, 0))
            if group_budget <= 0:
                continue
            if group_budget >= group_capacity[group_name]:
                for index in indices:
                    keep_masks[index] = torch.ones_like(normalized_scores[index])
                continue

            group_scores = [normalized_scores[index] for index in indices]
            flat_scores = torch.cat(group_scores, dim=0)
            keep_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            keep_idx = flat_scores.topk(group_budget, largest=True, sorted=False).indices
            keep_mask[keep_idx] = True

            offset = 0
            for index in indices:
                layer = layers[index]
                next_offset = offset + layer.r_max
                keep_masks[index] = keep_mask[offset:next_offset].to(
                    dtype=layer.rank_mask.dtype
                )
                offset = next_offset
        return keep_masks, group_budgets

    @staticmethod
    def _apply_rank_masks(layers, masks):
        for layer, mask in zip(layers, masks):
            layer.set_rank_mask(mask)

    @torch.no_grad()
    def _update_lora_rank_budget_with_counterfactual(
        self,
        layers,
        normalized_scores,
        keep_masks,
        total_budget,
        eval_metric_fn,
    ):
        current_masks = [layer.get_rank_mask() for layer in layers]
        candidate_records = []
        candidate_masks = []
        tested_masks = []
        safe_masks = []
        for index, (layer, current_mask, keep_mask, scores) in enumerate(
            zip(layers, current_masks, keep_masks, normalized_scores)
        ):
            candidate_mask = current_mask.gt(0) & keep_mask.le(0)
            candidate_masks.append(candidate_mask)
            tested_masks.append(torch.zeros_like(candidate_mask, dtype=torch.bool))
            safe_masks.append(torch.zeros_like(candidate_mask, dtype=torch.bool))
            candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
            for rank_index in candidate_indices.tolist():
                candidate_records.append(
                    (
                        float(scores[rank_index].item()),
                        index,
                        int(rank_index),
                    )
                )

        candidate_records.sort(key=lambda item: item[0])
        candidate_records = candidate_records[: self.lora_search_counterfactual_max_candidates]
        baseline_eval = eval_metric_fn()
        baseline_score = float(baseline_eval["score"])
        metric_name = baseline_eval.get("metric_name", "score")

        for _, layer_index, rank_index in candidate_records:
            layer = layers[layer_index]
            temp_mask = current_masks[layer_index].clone()
            temp_mask[rank_index] = 0.0
            layer.set_rank_mask(temp_mask)
            drop_eval = eval_metric_fn()
            drop_score = float(drop_eval["score"])
            layer.set_rank_mask(current_masks[layer_index])

            delta = drop_score - baseline_score
            tested_masks[layer_index][rank_index] = True
            if delta >= -self.lora_search_counterfactual_delta:
                safe_masks[layer_index][rank_index] = True

        final_masks = [mask.clone() for mask in current_masks]
        confirmed_pruned = 0
        tested_candidates = len(candidate_records)
        accepted_candidates = 0
        for index, layer in enumerate(layers):
            layer.update_counterfactual_confirm(
                candidate_masks[index],
                tested_masks[index],
                safe_masks[index],
            )
            confirmed_mask = candidate_masks[index] & layer.counterfactual_confirm.ge(
                self.lora_search_counterfactual_patience
            )
            accepted_candidates += int(safe_masks[index].sum().item())
            confirmed_pruned += int(confirmed_mask.sum().item())
            final_masks[index][confirmed_mask] = 0.0

        self._apply_rank_masks(layers, final_masks)

        active_ranks = [int(layer.active_rank.item()) for layer in layers]
        group_active_ranks = {}
        for layer in layers:
            group_active_ranks[layer.group_name] = (
                group_active_ranks.get(layer.group_name, 0)
                + int(layer.active_rank.item())
            )

        return {
            "total_active_rank": int(sum(active_ranks)),
            "active_ranks": active_ranks,
            "group_active_ranks": group_active_ranks,
            "counterfactual_tested": tested_candidates,
            "counterfactual_accepted": accepted_candidates,
            "counterfactual_pruned": confirmed_pruned,
            "counterfactual_metric_name": metric_name,
            "counterfactual_baseline_score": baseline_score,
            "counterfactual_budget_gap": int(max(0, sum(active_ranks) - total_budget)),
        }

    @torch.no_grad()
    def update_lora_rank_budget(self, budget_rank: int):
        layers = list(self.iter_searchable_lora_layers())
        if not layers:
            return None

        budget_rank, total_capacity, total_budget = self._compute_rank_budget(
            layers,
            budget_rank,
        )

        if total_budget <= 0:
            for layer in layers:
                layer.set_active_rank(0)
        elif total_budget >= total_capacity:
            for layer in layers:
                layer.set_active_rank(layer.r_max)
        else:
            normalized_scores, group_to_indices, group_capacity = self._compute_normalized_scores(layers)
            keep_masks, _ = self._build_keep_masks(
                layers,
                normalized_scores,
                total_budget,
                total_capacity,
                group_to_indices,
                group_capacity,
            )
            self._apply_rank_masks(layers, keep_masks)

        active_ranks = [int(layer.active_rank.item()) for layer in layers]
        group_active_ranks = {}
        for layer in layers:
            group_active_ranks[layer.group_name] = (
                group_active_ranks.get(layer.group_name, 0)
                + int(layer.active_rank.item())
            )
        return {
            "budget_rank": budget_rank,
            "total_active_rank": int(sum(active_ranks)),
            "active_ranks": active_ranks,
            "group_active_ranks": group_active_ranks,
            "budget_mode": self.lora_search_budget_mode,
        }

    @torch.no_grad()
    def update_lora_rank_search(self, epoch: int, total_epochs: int, eval_metric_fn=None):
        if not self.lora_search or epoch % self.lora_search_interval != 0:
            return None

        if epoch <= self.lora_search_warmup_epochs:
            return self.update_lora_rank_budget(self.lora_r)

        decay_steps = max(1, total_epochs - self.lora_search_warmup_epochs - 1)
        progress = min(
            1.0,
            max(0.0, float(epoch - self.lora_search_warmup_epochs - 1) / decay_steps),
        )
        budget_rank = int(
            round(self.lora_r + (self.lora_r_target - self.lora_r) * progress)
        )
        layers = list(self.iter_searchable_lora_layers())
        if not layers:
            return None

        budget_rank, total_capacity, total_budget = self._compute_rank_budget(
            layers,
            budget_rank,
        )
        if (
            not self.lora_search_counterfactual
            or eval_metric_fn is None
            or total_budget <= 0
            or total_budget >= total_capacity
        ):
            return self.update_lora_rank_budget(budget_rank)

        normalized_scores, group_to_indices, group_capacity = self._compute_normalized_scores(layers)
        keep_masks, group_budgets = self._build_keep_masks(
            layers,
            normalized_scores,
            total_budget,
            total_capacity,
            group_to_indices,
            group_capacity,
        )
        summary = self._update_lora_rank_budget_with_counterfactual(
            layers,
            normalized_scores,
            keep_masks,
            total_budget,
            eval_metric_fn,
        )
        summary.update(
            {
                "budget_rank": budget_rank,
                "budget_mode": self.lora_search_budget_mode,
                "counterfactual": True,
                "target_total_budget": int(total_budget),
                "group_budgets": group_budgets,
            }
        )
        return summary

    def forward(self, x):
        scale_factor = 2 / (512 / x.shape[-1])
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )

        use_grad = self.training and self.use_lora

        with torch.set_grad_enabled(use_grad):
            feats = self.model.get_intermediate_layers(
                x, n=self.extract_ids, reshape=True, norm=True
            )
            feats_ = []
            for feat in feats:
                feats_.append(
                    F.interpolate(
                        feat,
                        scale_factor=scale_factor,
                        mode="bilinear",
                    )
                )
        return feats_


class SepAdapterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 64, act=nn.SiLU):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, r, kernel_size=1, bias=False),
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                r, r, kernel_size=3, padding=1, groups=r, bias=False
            ),
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.proj = nn.Conv2d(r, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        x = self.proj(x)
        return x


class DenseAdapterLite(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        bottleneck=64,
        share=False,
    ):
        super().__init__()
        if share:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck)]
            )
        else:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck) for _ in range(4)]
            )
        self.share = share

    def forward(self, feats):
        outs = []
        for i, x in enumerate(feats):
            x = F.interpolate(
                x,
                scale_factor=2 / (2**i),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            block = self.blocks[0] if self.share else self.blocks[i]
            outs.append(block(x))
        return outs
