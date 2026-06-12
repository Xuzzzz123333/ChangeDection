import torch
import torch.nn as nn


def _resolve_group_count(channels: int, preferred_groups: int) -> int:
    groups = max(1, min(preferred_groups, channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class IdentityChangePrior(nn.Module):
    def forward(self, feat1, feat2):
        return torch.abs(feat1 - feat2)


class AdaptiveChangePrior(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mode: str = "full",
        norm_groups: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if mode not in {"full", "lite"}:
            raise ValueError(f"Unsupported ACPC mode: {mode}")

        feat_groups = _resolve_group_count(dim, norm_groups)
        hidden_groups = _resolve_group_count(hidden_dim, norm_groups)

        self.norm = nn.GroupNorm(feat_groups, dim)
        self.expand = nn.Sequential(
            nn.Conv2d(dim * 3, hidden_dim, 1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_dim),
            nn.SiLU(inplace=True),
        )
        if mode == "full":
            self.context = nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.GroupNorm(hidden_groups, hidden_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.context = nn.Identity()

        self.project = nn.Conv2d(hidden_dim, dim, 1, bias=True)
        nn.init.normal_(self.project.weight, std=1e-3)
        nn.init.zeros_(self.project.bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, feat1, feat2):
        base = torch.abs(feat1 - feat2)
        norm1 = self.norm(feat1)
        norm2 = self.norm(feat2)
        relation = torch.cat(
            [norm1 + norm2, torch.abs(norm1 - norm2), norm1 * norm2], dim=1
        )
        delta = self.expand(relation)
        delta = self.context(delta)
        delta = self.project(delta)
    
        scale = 1.0 + self.residual_scale * torch.tanh(delta)
        return base * scale


class AdaptiveChangePriorPyramid(nn.Module):
    def __init__(
        self,
        dim: int,
        stage_modes=("full", "lite", "none", "none"),
        hidden_ratio: float = 0.5,
        norm_groups: int = 8,
        residual_scale: float = 0.05,
    ):
        super().__init__()
        if len(stage_modes) != 4:
            raise ValueError(
                f"AdaptiveChangePriorPyramid expects 4 stage modes for p2-p5, got {len(stage_modes)}."
            )

        hidden_dim = max(1, int(round(dim * hidden_ratio)))
        blocks = []
        for mode in stage_modes:
            if mode == "none":
                blocks.append(IdentityChangePrior())
            else:
                blocks.append(
                    AdaptiveChangePrior(
                        dim=dim,
                        hidden_dim=hidden_dim,
                        mode=mode,
                        norm_groups=norm_groups,
                        residual_scale=residual_scale,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, feats1, feats2):
        priors = []
        for block, feat1, feat2 in zip(self.blocks, feats1, feats2):
            priors.append(block(feat1, feat2))
        return tuple(priors)
