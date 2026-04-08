import torch
import torch.nn as nn


def _resolve_group_count(channels: int, preferred_groups: int) -> int:
    groups = max(1, min(preferred_groups, channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class IdentityPairInteraction(nn.Module):
    def forward(self, feat1, feat2):
        return feat1, feat2


class PairInteractionBlock(nn.Module):
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
            raise ValueError(f"Unsupported pair interaction mode: {mode}")

        feat_groups = _resolve_group_count(dim, norm_groups)
        hidden_groups = _resolve_group_count(hidden_dim, norm_groups)

        self.norm1 = nn.GroupNorm(feat_groups, dim)
        self.norm2 = nn.GroupNorm(feat_groups, dim)
        self.relation_proj = nn.Sequential(
            nn.Conv2d(dim * 4, hidden_dim, 1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_dim),
            nn.SiLU(inplace=True),##投影
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
            )##多一个3x3卷积来看周围
        else:
            self.context = nn.Identity()
          ##内容
        self.delta = nn.Conv2d(hidden_dim, 2 * dim, 1, bias=False)
       ##强度
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, 2 * dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, 2 * dim, 1, 1), residual_scale, dtype=torch.float32)
        )

    def forward(self, feat1, feat2):
        norm1 = self.norm1(feat1)
        norm2 = self.norm2(feat2)
        relation = torch.cat(
            [norm1, norm2, torch.abs(norm1 - norm2), norm1 * norm2], dim=1
        )
        relation = self.relation_proj(relation)##投影压缩
        relation = self.context(relation)##内容增强

        update = self.residual_scale * self.gate(relation) * self.delta(relation)##更新量
        update1, update2 = update.chunk(2, dim=1)
        return feat1 + update1, feat2 + update2


class PairLocalPyramid(nn.Module):
    def __init__(
        self,
        dim: int,
        stage_modes=("full", "full", "lite", "lite"),
        hidden_ratio: float = 1.0,
        norm_groups: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if len(stage_modes) != 4:
            raise ValueError(
                f"PairLocalPyramid expects 4 stage modes for p2-p5, got {len(stage_modes)}."
            )

        hidden_dim = max(1, int(round(dim * hidden_ratio)))
        blocks = []
        for mode in stage_modes:
            if mode == "none":
                blocks.append(IdentityPairInteraction())
            else:
                blocks.append(
                    PairInteractionBlock(
                        dim=dim,
                        hidden_dim=hidden_dim,
                        mode=mode,
                        norm_groups=norm_groups,
                        residual_scale=residual_scale,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, feats1, feats2):
        out1 = []
        out2 = []
        for block, feat1, feat2 in zip(self.blocks, feats1, feats2):
            next1, next2 = block(feat1, feat2)
            out1.append(next1)
            out2.append(next2)
        return tuple(out1), tuple(out2)
