import torch
import torch.nn as nn

from .mfce import RFConv2d


def _resolve_group_count(channels: int, preferred_groups: int) -> int:
    groups = max(1, min(preferred_groups, channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class IdentityPairInteraction(nn.Module):
    def forward(self, feat1, feat2):
        return feat1, feat2


class RFPairContext(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        hidden_groups: int,
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
        self.conv = RFConv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=0,
            dilation=1,
            groups=hidden_dim,
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
        self.norm = nn.GroupNorm(hidden_groups, hidden_dim)
        self.act = nn.SiLU(inplace=True)

    def rf_state(self):
        return self.conv.rf_state()

    def configure_search_schedule(self, **kwargs):
        return self.conv.configure_search_schedule(**kwargs)

    def merge_branches_(self):
        return self.conv.merge_branches_()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PairInteractionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mode: str = "full",
        norm_groups: int = 8,
        residual_scale: float = 0.1,
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
        if mode not in {"full", "lite"}:
            raise ValueError(f"Unsupported pair interaction mode: {mode}")

        feat_groups = _resolve_group_count(dim, norm_groups)
        hidden_groups = _resolve_group_count(hidden_dim, norm_groups)

        self.norm1 = nn.GroupNorm(feat_groups, dim)
        self.norm2 = nn.GroupNorm(feat_groups, dim)
        self.relation_proj = nn.Sequential(
            nn.Conv2d(dim * 4, hidden_dim, 1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.rf_enable = bool(rf_enable and mode == "full")
        if mode == "full":
            if self.rf_enable:
                self.context = RFPairContext(
                    hidden_dim=hidden_dim,
                    hidden_groups=hidden_groups,
                    rf_mode=rf_mode,
                    rf_num_branches=rf_num_branches,
                    rf_expand_rate=rf_expand_rate,
                    rf_min_dilation=rf_min_dilation,
                    rf_max_dilation=rf_max_dilation,
                    rf_search_interval=rf_search_interval,
                    rf_max_search_step=rf_max_search_step,
                    rf_init_weight=rf_init_weight,
                )
            else:
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
        self.delta = nn.Conv2d(hidden_dim, 2 * dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, 2 * dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, 2 * dim, 1, 1), residual_scale, dtype=torch.float32)
        )

    def rf_state(self):
        if not self.rf_enable:
            return {}
        return self.context.rf_state()

    def configure_rf_search(self, **kwargs):
        if not self.rf_enable:
            return None
        return self.context.configure_search_schedule(**kwargs)

    def merge_rf_branches_(self):
        if not self.rf_enable:
            return None
        return self.context.merge_branches_()

    def forward(self, feat1, feat2):
        norm1 = self.norm1(feat1)
        norm2 = self.norm2(feat2)
        relation = torch.cat(
            [norm1, norm2, torch.abs(norm1 - norm2), norm1 * norm2], dim=1
        )
        relation = self.relation_proj(relation)
        relation = self.context(relation)

        update = self.residual_scale * self.gate(relation) * self.delta(relation)
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
        rf_enable: bool = False,
        rf_mode: str = "rfsearch",
        rf_num_branches: int = 3,
        rf_expand_rate: float = 0.5,
        rf_min_dilation: int = 1,
        rf_max_dilations=None,
        rf_search_interval: int = 100,
        rf_max_search_step: int = 8,
        rf_init_weight: float = 0.01,
    ):
        super().__init__()
        if len(stage_modes) != 4:
            raise ValueError(
                f"PairLocalPyramid expects 4 stage modes for p2-p5, got {len(stage_modes)}."
            )

        self.rf_enable = bool(rf_enable)
        if rf_max_dilations is None:
            rf_max_dilations = [None] * len(stage_modes)
        elif len(rf_max_dilations) != len(stage_modes):
            raise ValueError(
                "PairLocalPyramid expects rf_max_dilations to match the stage count."
            )

        hidden_dim = max(1, int(round(dim * hidden_ratio)))
        blocks = []
        for mode, rf_max_dilation in zip(stage_modes, rf_max_dilations):
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
                        rf_enable=self.rf_enable,
                        rf_mode=rf_mode,
                        rf_num_branches=rf_num_branches,
                        rf_expand_rate=rf_expand_rate,
                        rf_min_dilation=rf_min_dilation,
                        rf_max_dilation=rf_max_dilation,
                        rf_search_interval=rf_search_interval,
                        rf_max_search_step=rf_max_search_step,
                        rf_init_weight=rf_init_weight,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def _iter_rf_blocks(self):
        stage_names = ("p2", "p3", "p4", "p5")
        if not self.rf_enable:
            return []
        return [
            (stage_name, block)
            for stage_name, block in zip(stage_names, self.blocks)
            if hasattr(block, "rf_state") and block.rf_state()
        ]

    def rf_states(self):
        return [
            {"name": stage_name, **block.rf_state()}
            for stage_name, block in self._iter_rf_blocks()
        ]

    def configure_rf_search(self, **kwargs):
        stage_summaries = []
        for stage_name, block in self._iter_rf_blocks():
            summary = block.configure_rf_search(**kwargs)
            if summary is not None:
                stage_summaries.append({"name": stage_name, **summary})
        if not stage_summaries:
            return None
        summary = stage_summaries[0].copy()
        summary["stages"] = stage_summaries
        return summary

    def merge_rf_branches_(self):
        return [
            {"name": stage_name, **block.merge_rf_branches_()}
            for stage_name, block in self._iter_rf_blocks()
        ]

    def forward(self, feats1, feats2):
        out1 = []
        out2 = []
        for block, feat1, feat2 in zip(self.blocks, feats1, feats2):
            next1, next2 = block(feat1, feat2)
            out1.append(next1)
            out2.append(next2)
        return tuple(out1), tuple(out2)
