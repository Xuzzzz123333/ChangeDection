import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALinear

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
    ):
        super().__init__()
        self.device = device
        self.use_lora = use_lora
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
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

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
    def inject_lora(module, prefix="", r=4, alpha=16, dropout=0.05):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and DINOV3Wrapper.should_apply_lora(full_name):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            else:
                DINOV3Wrapper.inject_lora(child, full_name, r=r, alpha=alpha, dropout=dropout)

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
