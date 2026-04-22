import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mfce import RFConv2d
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
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        rf_enable: bool = False,
        rf_mode: str = "rfsearch",
        rf_num_branches: int = 3,
        rf_expand_rate: float = 0.5,
        rf_min_dilation: int = 1,
        rf_max_dilation=None,
        rf_search_interval: int = 100,
        rf_max_search_step: int = 8,
        rf_init_weight: float = 0.01,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("DINOConvTokenBranch expects a positive odd kernel size.")
        self.rf_enable = bool(rf_enable)
        if self.rf_enable:
            self.depthwise = RFConv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=0,
                dilation=1,
                groups=dim,
                bias=False,
                num_branches=rf_num_branches,
                expand_rate=rf_expand_rate,
                min_dilation=rf_min_dilation,
                max_dilation=rf_max_dilation,
                init_weight=rf_init_weight,
                search_interval=rf_search_interval,
                max_search_step=rf_max_search_step,
                rf_mode=rf_mode,
            )
        else:
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

    def rf_state(self):
        if not self.rf_enable:
            return {}
        return self.depthwise.rf_state()

    def configure_rf_search(self, **kwargs):
        if not self.rf_enable:
            return None
        return self.depthwise.configure_search_schedule(**kwargs)

    def merge_rf_branches_(self):
        if not self.rf_enable:
            return None
        return self.depthwise.merge_branches_()

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
        rf_enable: bool = False,
        rf_mode: str = "rfsearch",
        rf_num_branches: int = 3,
        rf_expand_rate: float = 0.5,
        rf_min_dilation: int = 1,
        rf_max_dilation=None,
        rf_search_interval: int = 100,
        rf_max_search_step: int = 8,
        rf_init_weight: float = 0.01,
    ):
        super().__init__()
        self.block = block
        self.num_prefix_tokens = int(max(0, num_prefix_tokens))
        self.norm = nn.LayerNorm(dim)
        self.local_branch = DINOConvTokenBranch(
            dim=dim,
            kernel_size=kernel_size,
            rf_enable=rf_enable,
            rf_mode=rf_mode,
            rf_num_branches=rf_num_branches,
            rf_expand_rate=rf_expand_rate,
            rf_min_dilation=rf_min_dilation,
            rf_max_dilation=rf_max_dilation,
            rf_search_interval=rf_search_interval,
            rf_max_search_step=rf_max_search_step,
            rf_init_weight=rf_init_weight,
        )
        self.gamma = nn.Parameter(torch.full((dim,), float(init_scale)))

    def rf_state(self):
        return self.local_branch.rf_state()

    def configure_rf_search(self, **kwargs):
        return self.local_branch.configure_rf_search(**kwargs)

    def merge_rf_branches_(self):
        return self.local_branch.merge_rf_branches_()

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
        lora_search_strategy="classic",
        lora_search_probe_batches=5,
        lora_search_probe_refresh_interval=10,
        lora_search_probe_score_norm="zscore",
        lora_search_probe_keep_ratio=0.5,
        lora_search_probe_module_keep_ratio=0.75,
        lora_search_rf_delta=2,
        lora_search_rf_temperature=1.0,
        lora_search_counterfactual=False,
        lora_search_counterfactual_val_batches=2,
        lora_search_counterfactual_max_candidates=64,
        lora_search_counterfactual_delta=0.0,
        lora_search_counterfactual_patience=1,
        local_conv_enable=False,
        local_conv_blocks=(5, 11, 17, 23),
        local_conv_kernel_size=3,
        local_conv_init_scale=0.0,
        local_conv_rf_enable=False,
        local_conv_rf_mode="rfsearch",
        local_conv_rf_num_branches=3,
        local_conv_rf_expand_rate=0.5,
        local_conv_rf_min_dilation=1,
        local_conv_rf_max_dilations=None,
        local_conv_rf_search_interval=100,
        local_conv_rf_max_search_step=8,
        local_conv_rf_init_weight=0.01,
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
        self.lora_search_strategy = lora_search_strategy
        self.lora_search_probe_batches = int(max(0, lora_search_probe_batches))
        self.lora_search_probe_refresh_interval = int(
            max(0, lora_search_probe_refresh_interval)
        )
        self.lora_search_probe_score_norm = lora_search_probe_score_norm
        self.lora_search_probe_keep_ratio = float(lora_search_probe_keep_ratio)
        self.lora_search_probe_module_keep_ratio = float(
            lora_search_probe_module_keep_ratio
        )
        self.lora_search_rf_delta = int(max(0, lora_search_rf_delta))
        self.lora_search_rf_temperature = float(max(1e-6, lora_search_rf_temperature))
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
        self.local_conv_rf_enable = bool(local_conv_rf_enable and self.local_conv_enable)
        self.local_conv_rf_mode = local_conv_rf_mode
        self.local_conv_rf_num_branches = int(max(1, local_conv_rf_num_branches))
        self.local_conv_rf_expand_rate = float(local_conv_rf_expand_rate)
        self.local_conv_rf_min_dilation = int(max(1, local_conv_rf_min_dilation))
        self.local_conv_rf_max_dilations = local_conv_rf_max_dilations
        self.local_conv_rf_search_interval = int(max(1, local_conv_rf_search_interval))
        self.local_conv_rf_max_search_step = int(max(0, local_conv_rf_max_search_step))
        self.local_conv_rf_init_weight = float(local_conv_rf_init_weight)
        self.lora_rf_probe_ready = False
        self.lora_rf_selected_blocks = tuple()
        self.lora_rf_probe_epoch = -1
        self.lora_rf_probe_runs = 0
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
            rf_max_dilation = None
            if self.local_conv_rf_max_dilations is not None:
                if len(self.local_conv_rf_max_dilations) == 1:
                    rf_max_dilation = self.local_conv_rf_max_dilations[0]
                else:
                    try:
                        block_pos = valid_indices.index(block_index)
                    except ValueError:
                        block_pos = 0
                    rf_max_dilation = self.local_conv_rf_max_dilations[block_pos]
            blocks[block_index] = DINOBlockLocalConvAdapter(
                block=block,
                dim=dim,
                num_prefix_tokens=num_prefix_tokens,
                kernel_size=kernel_size,
                init_scale=init_scale,
                rf_enable=self.local_conv_rf_enable,
                rf_mode=self.local_conv_rf_mode,
                rf_num_branches=self.local_conv_rf_num_branches,
                rf_expand_rate=self.local_conv_rf_expand_rate,
                rf_min_dilation=self.local_conv_rf_min_dilation,
                rf_max_dilation=rf_max_dilation,
                rf_search_interval=self.local_conv_rf_search_interval,
                rf_max_search_step=self.local_conv_rf_max_search_step,
                rf_init_weight=self.local_conv_rf_init_weight,
            ).to(self.device)

    def _iter_local_conv_rf_blocks(self):
        if not self.local_conv_rf_enable:
            return []
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            return []
        return [
            (f"block{block_index}", blocks[block_index])
            for block_index in self.local_conv_blocks
            if block_index < len(blocks)
            and isinstance(blocks[block_index], DINOBlockLocalConvAdapter)
            and blocks[block_index].rf_state()
        ]

    def local_conv_rf_states(self):
        return [
            {"name": name, **module.rf_state()}
            for name, module in self._iter_local_conv_rf_blocks()
        ]

    def configure_local_conv_rf_search(self, **kwargs):
        stage_summaries = []
        for name, module in self._iter_local_conv_rf_blocks():
            summary = module.configure_rf_search(**kwargs)
            if summary is not None:
                stage_summaries.append({"name": name, **summary})
        if not stage_summaries:
            return None
        summary = stage_summaries[0].copy()
        summary["stages"] = stage_summaries
        return summary

    def merge_local_conv_rf_branches_(self):
        return [
            {"name": name, **module.merge_rf_branches_()}
            for name, module in self._iter_local_conv_rf_blocks()
        ]

    @staticmethod
    def _extract_block_index_from_module_name(module_name: str):
        return DINOV3Wrapper._extract_layer_index(module_name)

    def _allocate_weighted_budgets(self, total_budget: int, capacities: dict, weights: dict):
        if total_budget <= 0:
            return {key: 0 for key in capacities}

        weighted_capacity = {}
        for key, capacity in capacities.items():
            weighted_capacity[key] = capacity * max(0.0, float(weights.get(key, 0.0)))

        if sum(weighted_capacity.values()) <= 0:
            weighted_capacity = {key: float(capacity) for key, capacity in capacities.items()}

        raw_budget = {
            key: total_budget * weighted_capacity[key] / sum(weighted_capacity.values())
            for key in capacities
        }
        allocated = {
            key: min(capacities[key], int(raw_budget[key]))
            for key in capacities
        }
        remaining = total_budget - sum(allocated.values())

        while remaining > 0:
            candidates = [
                key for key, capacity in capacities.items() if allocated[key] < capacity
            ]
            if not candidates:
                break
            candidates.sort(
                key=lambda key: (
                    raw_budget[key] - int(raw_budget[key]),
                    float(weights.get(key, 0.0)),
                    capacities[key] - allocated[key],
                ),
                reverse=True,
            )
            allocated[candidates[0]] += 1
            remaining -= 1
        return allocated

    @staticmethod
    def _layer_rank_utility(scores: torch.Tensor, rank: int) -> float:
        rank = int(max(0, min(rank, scores.numel())))
        if rank <= 0:
            return 0.0
        top_scores = torch.topk(scores, rank, largest=True, sorted=False).values
        return float(top_scores.sum().item())

    @staticmethod
    def _normalize_score_map(score_map: dict, mode: str):
        if not score_map:
            return {}

        keys = list(score_map.keys())
        values = torch.tensor(
            [float(score_map[key]) for key in keys],
            dtype=torch.float32,
        )
        if mode == "median":
            values = values / values.median().clamp_min(1e-6)
        elif mode == "zscore":
            if values.numel() > 1:
                values = (values - values.mean()) / values.std(unbiased=False).clamp_min(
                    1e-6
                )
            else:
                values = values - values.mean()
        return {key: float(value.item()) for key, value in zip(keys, values)}

    @staticmethod
    def _quantile_map(score_map: dict):
        if not score_map:
            return {}

        items = sorted(
            score_map.items(),
            key=lambda item: (float(item[1]), str(item[0])),
        )
        if len(items) == 1:
            return {items[0][0]: 1.0}

        mapped = {}
        index = 0
        while index < len(items):
            next_index = index
            current_value = float(items[index][1])
            while (
                next_index + 1 < len(items)
                and abs(float(items[next_index + 1][1]) - current_value) <= 1e-12
            ):
                next_index += 1
            quantile = 0.5 * (index + next_index) / float(len(items) - 1)
            for item_index in range(index, next_index + 1):
                mapped[items[item_index][0]] = quantile
            index = next_index + 1
        return mapped

    def _build_rfnext_probe_state(self, layers, probe_scores: dict):
        module_raw_scores = {
            layer.module_name: float(probe_scores.get(layer.module_name, 0.0))
            for layer in layers
        }
        module_norm_scores = self._normalize_score_map(
            module_raw_scores,
            self.lora_search_probe_score_norm,
        )
        module_quantiles = self._quantile_map(module_norm_scores)

        block_to_modules = {}
        layer_block_indices = {}
        for layer in layers:
            block_index = self._extract_block_index_from_module_name(layer.module_name)
            layer_block_indices[layer.module_name] = block_index
            if block_index is None:
                continue
            block_to_modules.setdefault(block_index, []).append(layer.module_name)

        block_raw_scores = {}
        bucket_best_blocks = {}
        for block_index, module_names in block_to_modules.items():
            block_raw_scores[block_index] = sum(
                module_norm_scores[name] for name in module_names
            ) / max(1, len(module_names))
            bucket_label = self._get_depth_bucket_label(
                block_index,
                self.n_layers,
                self.lora_search_depth_buckets,
            )
            current_best = bucket_best_blocks.get(bucket_label)
            if (
                current_best is None
                or block_raw_scores[block_index] > block_raw_scores[current_best]
            ):
                bucket_best_blocks[bucket_label] = block_index

        block_quantiles = self._quantile_map(block_raw_scores)
        sorted_blocks = sorted(
            block_raw_scores,
            key=lambda block_index: (
                block_quantiles.get(block_index, 0.0),
                block_raw_scores[block_index],
                -block_index,
            ),
            reverse=True,
        )
        keep_count = min(
            len(sorted_blocks),
            max(
                1,
                int(math.ceil(len(sorted_blocks) * self.lora_search_probe_keep_ratio)),
            ),
        )
        selected_blocks = set(sorted_blocks[:keep_count])
        selected_blocks.update(bucket_best_blocks.values())

        selected_modules = set()
        for block_index in selected_blocks:
            module_names = block_to_modules.get(block_index, [])
            if not module_names:
                continue
            module_keep_count = min(
                len(module_names),
                max(
                    1,
                    int(
                        math.ceil(
                            len(module_names)
                            * self.lora_search_probe_module_keep_ratio
                        )
                    ),
                ),
            )
            ranked_modules = sorted(
                module_names,
                key=lambda module_name: (
                    module_quantiles.get(module_name, 0.0),
                    module_norm_scores.get(module_name, 0.0),
                    module_name,
                ),
                reverse=True,
            )
            selected_modules.update(ranked_modules[:module_keep_count])

        module_residuals = {}
        for block_index, module_names in block_to_modules.items():
            block_module_scores = {
                module_name: module_quantiles.get(module_name, 0.0)
                for module_name in module_names
            }
            module_residuals.update(self._quantile_map(block_module_scores))

        module_state = {}
        for layer in layers:
            module_name = layer.module_name
            block_index = layer_block_indices[module_name]
            block_score = (
                float(block_quantiles.get(block_index, 0.0))
                if block_index is not None
                else 0.0
            )
            module_score = float(module_quantiles.get(module_name, 0.0))
            module_residual = float(module_residuals.get(module_name, 0.0))
            selected = (
                block_index in selected_blocks and module_name in selected_modules
                if block_index is not None
                else False
            )
            rank_prior = (0.7 * block_score + 0.3 * module_residual) if selected else 0.0
            module_state[module_name] = {
                "block_index": block_index,
                "module_score": module_score,
                "block_score": block_score,
                "module_residual": module_residual,
                "rank_prior": rank_prior,
                "selected": selected,
            }

        return {
            "module_state": module_state,
            "selected_blocks": tuple(sorted(selected_blocks)),
            "selected_modules": tuple(sorted(selected_modules)),
        }

    def _allocate_rfnext_hierarchical_budgets(
        self,
        layers,
        total_budget: int,
        module_weights: dict,
    ):
        layer_budgets = {layer.module_name: 0 for layer in layers}
        selected_layers = [layer for layer in layers if bool(layer.probe_selected.item())]
        if total_budget <= 0 or not selected_layers:
            return layer_budgets

        block_to_layers = {}
        block_capacities = {}
        block_weights = {}
        for layer in selected_layers:
            block_index = self._extract_block_index_from_module_name(layer.module_name)
            if block_index is None:
                continue
            block_to_layers.setdefault(block_index, []).append(layer)
            block_capacities[block_index] = block_capacities.get(block_index, 0) + layer.r_max
            block_weights[block_index] = block_weights.get(block_index, 0.0) + max(
                0.0,
                float(module_weights.get(layer.module_name, 0.0)),
            )

        if not block_to_layers:
            return layer_budgets

        for block_index, block_layers in block_to_layers.items():
            if block_weights.get(block_index, 0.0) <= 0:
                block_weights[block_index] = max(
                    1e-6,
                    sum(float(layer.probe_block_score.item()) for layer in block_layers)
                    / max(1, len(block_layers)),
                )

        block_budgets = self._allocate_weighted_budgets(
            total_budget,
            block_capacities,
            block_weights,
        )
        for block_index, block_layers in block_to_layers.items():
            block_budget = int(block_budgets.get(block_index, 0))
            if block_budget <= 0:
                continue
            module_capacities = {
                layer.module_name: layer.r_max for layer in block_layers
            }
            module_weight_map = {
                layer.module_name: max(
                    0.0,
                    float(module_weights.get(layer.module_name, 0.0)),
                )
                for layer in block_layers
            }
            if sum(module_weight_map.values()) <= 0:
                for layer in block_layers:
                    module_weight_map[layer.module_name] = max(
                        1e-6,
                        0.7 * float(layer.probe_rank_prior.item())
                        + 0.3 * float(layer.probe_module_residual.item()),
                    )
            block_layer_budgets = self._allocate_weighted_budgets(
                block_budget,
                module_capacities,
                module_weight_map,
            )
            layer_budgets.update(block_layer_budgets)
        return layer_budgets

    def _refresh_rfnext_probe(
        self,
        layers,
        total_budget: int,
        probe_scores: dict,
        epoch: int,
    ):
        probe_state = self._build_rfnext_probe_state(layers, probe_scores)
        module_state = probe_state["module_state"]

        weights = {}
        selected_layers = []
        for layer in layers:
            state = module_state.get(
                layer.module_name,
                {
                    "module_score": 0.0,
                    "block_score": 0.0,
                    "module_residual": 0.0,
                    "rank_prior": 0.0,
                    "selected": False,
                },
            )
            layer.set_probe_score(state["module_score"])
            layer.set_probe_prior(
                state["block_score"],
                state["module_residual"],
                state["rank_prior"],
            )
            layer.set_probe_selected(state["selected"])
            weights[layer.module_name] = max(0.0, float(state["rank_prior"]))
            if state["selected"]:
                selected_layers.append(layer.module_name)

        if selected_layers and sum(weights[name] for name in selected_layers) <= 0:
            for module_name in selected_layers:
                weights[module_name] = 1.0

        layer_budgets = self._allocate_rfnext_hierarchical_budgets(
            layers,
            total_budget,
            weights,
        )
        is_refresh = self.lora_rf_probe_ready
        for layer in layers:
            is_selected = bool(layer.probe_selected.item())
            target_center = (
                float(layer_budgets.get(layer.module_name, 0))
                if is_selected
                else 0.0
            )
            if is_selected:
                # Smooth the first rfnext transition from the currently active rank
                # instead of hard-resetting centers from the full-rank/classic phase.
                previous_center = (
                    float(layer.rank_center.item())
                    if is_refresh
                    else float(layer.active_rank.item())
                )
                target_center = 0.5 * previous_center + 0.5 * target_center
            layer.set_rank_center(target_center)

        self.lora_rf_probe_ready = True
        self.lora_rf_selected_blocks = probe_state["selected_blocks"]
        self.lora_rf_probe_epoch = int(epoch)
        self.lora_rf_probe_runs += 1
        return {
            "probe_selected_blocks": len(self.lora_rf_selected_blocks),
            "probe_selected_layers": int(
                sum(int(layer.probe_selected.item()) for layer in layers)
            ),
            "probe_selected_modules": int(
                sum(int(layer.probe_selected.item()) for layer in layers)
            ),
            "probe_block_indices": self.lora_rf_selected_blocks,
            "probe_epoch": int(self.lora_rf_probe_epoch),
            "probe_refreshed": bool(is_refresh),
            "probe_runs": int(self.lora_rf_probe_runs),
        }

    def _adaptive_rf_delta(self, layer: SearchableLoRALinear):
        if self.lora_search_rf_delta <= 0:
            return 0
        prior = float(layer.probe_rank_prior.item())
        delta = int(round(self.lora_search_rf_delta * (0.5 + prior)))
        return max(1, min(layer.r_max, delta))

    def _local_rank_candidates(self, layer: SearchableLoRALinear):
        center_int = int(round(float(layer.rank_center.item())))
        candidates = {max(0, min(layer.r_max, center_int))}
        delta = self._adaptive_rf_delta(layer)
        if delta > 0:
            candidates.add(max(0, min(layer.r_max, center_int - delta)))
            candidates.add(max(0, min(layer.r_max, center_int + delta)))
        return sorted(candidates)

    def _allocate_rfnext_layer_budgets(self, layers, normalized_scores, total_budget):
        expected_weights = {}

        for layer, scores in zip(layers, normalized_scores):
            if not bool(layer.probe_selected.item()):
                expected_weights[layer.module_name] = 0.0
                layer.set_rank_center(0.0)
                continue

            candidates = self._local_rank_candidates(layer)
            utilities = torch.tensor(
                [self._layer_rank_utility(scores, rank) for rank in candidates],
                dtype=torch.float32,
                device=scores.device,
            )
            logits = utilities / self.lora_search_rf_temperature
            probs = torch.softmax(logits - logits.max(), dim=0)
            expected_rank = float(
                sum(rank * float(prob.item()) for rank, prob in zip(candidates, probs))
            )
            prior_rank = float(layer.probe_rank_prior.item()) * float(layer.r_max)
            refined_rank = 0.5 * expected_rank + 0.5 * prior_rank
            expected_weights[layer.module_name] = max(0.0, refined_rank)
            layer.set_rank_center(refined_rank)

        selected_layers = [
            layer.module_name for layer in layers if bool(layer.probe_selected.item())
        ]
        if selected_layers and sum(expected_weights[name] for name in selected_layers) <= 0:
            for layer in layers:
                if bool(layer.probe_selected.item()):
                    expected_weights[layer.module_name] = max(
                        1.0,
                        float(layer.probe_rank_prior.item()) * float(layer.r_max),
                    )

        return self._allocate_rfnext_hierarchical_budgets(
            layers,
            total_budget,
            expected_weights,
        )

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

    def _build_keep_masks_from_layer_budgets(self, layers, normalized_scores, layer_budgets):
        keep_masks = []
        for layer, scores in zip(layers, normalized_scores):
            layer_budget = int(max(0, min(layer.r_max, layer_budgets.get(layer.module_name, 0))))
            if layer_budget <= 0:
                keep_masks.append(torch.zeros_like(scores))
                continue
            if layer_budget >= layer.r_max:
                keep_masks.append(torch.ones_like(scores))
                continue
            keep_mask = torch.zeros_like(scores, dtype=torch.bool)
            keep_idx = scores.topk(layer_budget, largest=True, sorted=False).indices
            keep_mask[keep_idx] = True
            keep_masks.append(keep_mask.to(dtype=layer.rank_mask.dtype))
        return keep_masks

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
            "search_strategy": self.lora_search_strategy,
        }

    @torch.no_grad()
    def update_lora_rank_search(
        self,
        epoch: int,
        total_epochs: int,
        eval_metric_fn=None,
        probe_score_fn=None,
    ):
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
            self.lora_search_strategy == "rfnext"
            and probe_score_fn is None
            and not self.lora_rf_probe_ready
        ):
            summary = self.update_lora_rank_budget(budget_rank)
            if summary is not None:
                summary["search_strategy"] = "classic"
                summary["search_fallback"] = "missing_probe"
            return summary
        if self.lora_search_strategy == "rfnext" and total_budget > 0 and total_budget < total_capacity:
            probe_summary = {}
            should_refresh_probe = False
            if probe_score_fn is not None:
                if not self.lora_rf_probe_ready:
                    should_refresh_probe = True
                elif self.lora_search_probe_refresh_interval > 0:
                    should_refresh_probe = (
                        epoch - self.lora_rf_probe_epoch
                        >= self.lora_search_probe_refresh_interval
                    )
            if should_refresh_probe:
                with torch.enable_grad():
                    probe_scores = probe_score_fn()
                if probe_scores:
                    probe_summary = self._refresh_rfnext_probe(
                        layers,
                        total_budget,
                        probe_scores,
                        epoch,
                    )
            normalized_scores, group_to_indices, group_capacity = self._compute_normalized_scores(layers)
            layer_budgets = self._allocate_rfnext_layer_budgets(
                layers,
                normalized_scores,
                total_budget,
            )
            keep_masks = self._build_keep_masks_from_layer_budgets(
                layers,
                normalized_scores,
                layer_budgets,
            )
            if not self.lora_search_counterfactual or eval_metric_fn is None:
                self._apply_rank_masks(layers, keep_masks)
                active_ranks = [int(layer.active_rank.item()) for layer in layers]
                group_active_ranks = {}
                for layer in layers:
                    group_active_ranks[layer.group_name] = (
                        group_active_ranks.get(layer.group_name, 0)
                        + int(layer.active_rank.item())
                    )
                summary = {
                    "budget_rank": budget_rank,
                    "total_active_rank": int(sum(active_ranks)),
                    "active_ranks": active_ranks,
                    "group_active_ranks": group_active_ranks,
                    "budget_mode": self.lora_search_budget_mode,
                    "search_strategy": "rfnext",
                }
                summary.update(probe_summary)
                if not probe_summary and self.lora_rf_probe_ready:
                    summary.update(
                        {
                            "probe_selected_blocks": len(self.lora_rf_selected_blocks),
                            "probe_selected_layers": int(
                                sum(int(layer.probe_selected.item()) for layer in layers)
                            ),
                            "probe_selected_modules": int(
                                sum(int(layer.probe_selected.item()) for layer in layers)
                            ),
                            "probe_block_indices": self.lora_rf_selected_blocks,
                            "probe_epoch": int(self.lora_rf_probe_epoch),
                            "probe_refreshed": False,
                            "probe_runs": int(self.lora_rf_probe_runs),
                        }
                    )
                return summary

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
                    "group_budgets": {},
                    "search_strategy": "rfnext",
                }
            )
            summary.update(probe_summary)
            if not probe_summary and self.lora_rf_probe_ready:
                summary.update(
                    {
                        "probe_selected_blocks": len(self.lora_rf_selected_blocks),
                        "probe_selected_layers": int(
                            sum(int(layer.probe_selected.item()) for layer in layers)
                        ),
                        "probe_selected_modules": int(
                            sum(int(layer.probe_selected.item()) for layer in layers)
                        ),
                        "probe_block_indices": self.lora_rf_selected_blocks,
                        "probe_epoch": int(self.lora_rf_probe_epoch),
                        "probe_refreshed": False,
                        "probe_runs": int(self.lora_rf_probe_runs),
                    }
                )
            return summary
        if (
            not self.lora_search_counterfactual
            or eval_metric_fn is None
            or total_budget <= 0
            or total_budget >= total_capacity
        ):
            summary = self.update_lora_rank_budget(budget_rank)
            if summary is not None:
                summary["search_strategy"] = "classic"
            return summary

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
                "search_strategy": "classic",
            }
        )
        return summary

    def forward(self, x):
        scale_factor = 2 / (512 / x.shape[-1])
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )

        use_grad = self.training and (self.use_lora or self.local_conv_enable)

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
