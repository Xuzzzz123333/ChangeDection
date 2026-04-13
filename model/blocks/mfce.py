import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 1, act=nn.SiLU):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            act(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        act=nn.SiLU,
    ):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                in_dim,
                in_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_dim,
                bias=False,
            ),
            nn.BatchNorm2d(in_dim),
            act(inplace=True),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            act(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthAttnFuse(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.projs = nn.ModuleList(
            [ConvBnAct(in_dim, hidden_dim, kernel_size=1) for _ in range(num_layers)]
        )
        self.scores = nn.ModuleList(
            [nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True) for _ in range(num_layers)]
        )
        self.out_proj = ConvBnAct(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, feats):
        if len(feats) != len(self.projs):
            raise ValueError(
                f"DepthAttnFuse expects {len(self.projs)} features, got {len(feats)}."
            )

        proj_feats = []
        score_maps = []
        for feat, proj, score in zip(feats, self.projs, self.scores):
            proj_feat = proj(feat)
            proj_feats.append(proj_feat)
            score_maps.append(score(proj_feat))

        feat_stack = torch.stack(proj_feats, dim=1)
        score_stack = torch.stack(score_maps, dim=1)
        attn = torch.softmax(score_stack, dim=1)
        fused = (feat_stack * attn).sum(dim=1)
        return self.out_proj(fused)


class ASPPContext(nn.Module):
    def __init__(self, dim: int, rates=(1, 2, 4, 8)):
        super().__init__()
        if not rates:
            raise ValueError("ASPPContext expects at least one dilation rate.")

        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv(
                    dim,
                    dim,
                    kernel_size=3,
                    dilation=max(1, int(rate)),
                )
                for rate in rates
            ]
        )
        self.project = ConvBnAct(dim * len(self.branches), dim, kernel_size=1)

    def forward(self, x):
        branch_feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(branch_feats, dim=1))


class MFCEPyramidAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 256,
        mid_dim: int = 256,
        aspp_rates=(1, 2, 4, 8),
    ):
        super().__init__()
        self.depth_fuse = DepthAttnFuse(in_dim=in_dim, hidden_dim=mid_dim, num_layers=4)
        self.context = ASPPContext(mid_dim, rates=tuple(aspp_rates))
        self.refine_blocks = nn.ModuleList(
            [DepthwiseSeparableConv(mid_dim, mid_dim, kernel_size=3) for _ in range(4)]
        )
        self.out_blocks = nn.ModuleList(
            [nn.Conv2d(mid_dim, out_dim, kernel_size=1, bias=True) for _ in range(4)]
        )
        self.scales = (2.0, 1.0, 0.5, 0.25)

    def forward(self, feats):
        fused = self.depth_fuse(feats)
        fused = self.context(fused)

        outs = []
        for scale, refine, out_block in zip(self.scales, self.refine_blocks, self.out_blocks):
            if scale == 1.0:
                feat = fused
            else:
                feat = F.interpolate(
                    fused,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
            feat = refine(feat)
            outs.append(out_block(feat))
        return outs


class TemporalFeatureExchange(nn.Module):
    def __init__(
        self,
        mode: str = "layer",
        thresh: float = 0.5,
        p: int = 2,
        layers=(0, 1, 2, 3),
    ):
        super().__init__()
        valid_modes = {
            "none",
            "layer",
            "rand_layer",
            "channel",
            "rand_channel",
            "spatial",
            "rand_spatial",
        }
        if mode not in valid_modes:
            raise ValueError(f"Unsupported temporal exchange mode: {mode}")

        self.mode = mode
        self.thresh = float(thresh)
        self.p = max(1, int(p))
        self.layers = tuple(sorted(set(int(index) for index in layers)))

    @staticmethod
    def _swap_by_mask(feat1, feat2, mask):
        return torch.where(mask, feat2, feat1), torch.where(mask, feat1, feat2)

    def _channel_mask(self, feat):
        channels = feat.shape[1]
        if self.mode == "rand_channel":
            mask = torch.rand(channels, device=feat.device) < self.thresh
        else:
            mask = torch.zeros(channels, dtype=torch.bool, device=feat.device)
            mask[:: self.p] = True
        return mask.view(1, channels, 1, 1)

    def _spatial_mask(self, feat):
        height, width = feat.shape[-2:]
        if self.mode == "rand_spatial":
            mask = torch.rand(height, width, device=feat.device) < self.thresh
        else:
            mask = torch.zeros(height, width, dtype=torch.bool, device=feat.device)
            mask[:: self.p, :] = True
            mask[:, :: self.p] = True
        return mask.view(1, 1, height, width)

    def _should_swap_layer(self, layer_index: int):
        if layer_index not in self.layers:
            return False
        if self.mode == "layer":
            return layer_index % 2 == 0
        return torch.rand(1, device="cpu").item() < self.thresh

    def forward(self, feats1, feats2):
        if self.mode == "none":
            return tuple(feats1), tuple(feats2)
        if len(feats1) != len(feats2):
            raise ValueError(
                f"TemporalFeatureExchange expects paired feature lists, got {len(feats1)} and {len(feats2)}."
            )

        out1 = list(feats1)
        out2 = list(feats2)

        for layer_index, (feat1, feat2) in enumerate(zip(out1, out2)):
            if self.mode in {"layer", "rand_layer"}:
                if self._should_swap_layer(layer_index):
                    out1[layer_index], out2[layer_index] = feat2, feat1
                continue

            if layer_index not in self.layers:
                continue

            if self.mode in {"channel", "rand_channel"}:
                mask = self._channel_mask(feat1)
            else:
                mask = self._spatial_mask(feat1)

            out1[layer_index], out2[layer_index] = self._swap_by_mask(feat1, feat2, mask)

        return tuple(out1), tuple(out2)
