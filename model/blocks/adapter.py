import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALinear, SearchableLoRALinear

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


class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        extract_ids=[5, 11, 17, 23],
        device="cuda",
        use_lora=False,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_search=False,
        lora_r_target=None,
        lora_alpha_over_r=1.0,
        lora_search_warmup_epochs=5,
        lora_search_interval=1,
    ):
        super().__init__()
        self.device = device
        self.use_lora = use_lora
        self.lora_search = self.use_lora and lora_search
        self.lora_r = int(lora_r)
        self.lora_r_target = (
            self.lora_r if lora_r_target is None else int(max(0, lora_r_target))
        )
        self.lora_alpha_over_r = float(lora_alpha_over_r)
        self.lora_search_warmup_epochs = int(max(0, lora_search_warmup_epochs))
        self.lora_search_interval = int(max(1, lora_search_interval))
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

        if self.use_lora:
            self.inject_lora(
                self.model,
                r=self.lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                search=self.lora_search,
                alpha_over_r=self.lora_alpha_over_r,
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
    def inject_lora(
        module,
        prefix="",
        r=4,
        alpha=16,
        dropout=0.05,
        search=False,
        alpha_over_r=1.0,
    ):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and DINOV3Wrapper.should_apply_lora(full_name):
                if search:
                    setattr(
                        module,
                        name,
                        SearchableLoRALinear(
                            child,
                            r=r,
                            alpha_over_r=alpha_over_r,
                            dropout=dropout,
                        ),
                    )
                else:
                    setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            else:
                DINOV3Wrapper.inject_lora(
                    child,
                    full_name,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    search=search,
                    alpha_over_r=alpha_over_r,
                )

    def iter_searchable_lora_layers(self):
        for module in self.model.modules():
            if isinstance(module, SearchableLoRALinear):
                yield module

    @torch.no_grad()
    def update_lora_rank_budget(self, budget_rank: int):
        layers = list(self.iter_searchable_lora_layers())
        if not layers:
            return None

        budget_rank = int(max(0, min(budget_rank, self.lora_r)))
        total_capacity = sum(layer.r_max for layer in layers)
        total_budget = min(total_capacity, budget_rank * len(layers))

        if total_budget <= 0:
            for layer in layers:
                layer.set_active_rank(0)
        elif total_budget >= total_capacity:
            for layer in layers:
                layer.set_active_rank(layer.r_max)
        else:
            score_chunks = [layer.importance_scores() for layer in layers]
            flat_scores = torch.cat(score_chunks, dim=0)
            keep_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            keep_idx = flat_scores.topk(total_budget, largest=True, sorted=False).indices
            keep_mask[keep_idx] = True

            offset = 0
            for layer in layers:
                next_offset = offset + layer.r_max
                local_mask = keep_mask[offset:next_offset].to(dtype=layer.rank_mask.dtype)
                layer.set_rank_mask(local_mask)
                offset = next_offset

        active_ranks = [int(layer.active_rank.item()) for layer in layers]
        return {
            "budget_rank": budget_rank,
            "total_active_rank": int(sum(active_ranks)),
            "active_ranks": active_ranks,
        }

    @torch.no_grad()
    def update_lora_rank_search(self, epoch: int, total_epochs: int):
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
        return self.update_lora_rank_budget(budget_rank)

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
