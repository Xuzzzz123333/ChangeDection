import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import group_norm


def _to_2tuple(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple value, got {value}.")
        return (int(value[0]), int(value[1]))
    value = int(value)
    return (value, value)


def _same_padding(kernel_size: int, stride: int, dilation: int):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def value_crop(dilation, min_dilation, max_dilation):
    if min_dilation is not None and dilation < min_dilation:
        dilation = min_dilation
    if max_dilation is not None and dilation > max_dilation:
        dilation = max_dilation
    return dilation


def rf_expand(dilation, expand_rate, num_branches, min_dilation=1, max_dilation=None):
    if num_branches < 2:
        raise ValueError("rf_expand expects num_branches >= 2.")

    dilation = _to_2tuple(dilation)
    delta_h = expand_rate * dilation[0]
    delta_w = expand_rate * dilation[1]
    rate_list = []
    for index in range(num_branches):
        rate_list.append(
            (
                value_crop(
                    int(
                        round(
                            dilation[0]
                            - delta_h
                            + index * 2 * delta_h / (num_branches - 1)
                        )
                    ),
                    min_dilation,
                    max_dilation,
                ),
                value_crop(
                    int(
                        round(
                            dilation[1]
                            - delta_w
                            + index * 2 * delta_w / (num_branches - 1)
                        )
                    ),
                    min_dilation,
                    max_dilation,
                ),
            )
        )

    unique_rate_list = list(dict.fromkeys(rate_list))
    return unique_rate_list


class ConvBnAct(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 1, act=nn.SiLU):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding, bias=False),
            group_norm(out_dim),
            act(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RFConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        num_branches=3,
        expand_rate=0.5,
        min_dilation=1,
        max_dilation=None,
        init_weight=0.01,
        search_interval=100,
        max_search_step=8,
        rf_mode="rfsearch",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        if rf_mode not in {"rfsearch", "rfsingle", "rfmultiple", "rfmerge"}:
            raise ValueError(f"Unsupported RF mode: {rf_mode}")
        if num_branches <= 0:
            raise ValueError("RFConv2d expects num_branches > 0.")

        self.rf_mode = rf_mode
        self.num_branches = int(num_branches)
        self.max_dilation = None if max_dilation is None else int(max_dilation)
        self.min_dilation = max(1, int(min_dilation))
        self.expand_rate = float(expand_rate)
        self.init_weight = float(init_weight)
        self.search_interval = max(1, int(search_interval))
        self.max_search_step = max(0, int(max_search_step))

        self.sample_weights = nn.Parameter(torch.empty(self.num_branches))
        self.register_buffer("counter", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("forward_step", torch.zeros(1, dtype=torch.int64))
        self.register_buffer(
            "current_search_step", torch.zeros(1, dtype=torch.int64)
        )
        self.register_buffer("search_start_step", torch.zeros(1, dtype=torch.int64))
        self.register_buffer(
            "search_stop_step", torch.full((1,), -1, dtype=torch.int64)
        )
        self.register_buffer(
            "rates", torch.ones((self.num_branches, 2), dtype=torch.int32)
        )
        self.register_buffer("num_rates", torch.ones(1, dtype=torch.int32))
        self.register_buffer("merged", torch.tensor(False, dtype=torch.bool))

        init_dilation = _to_2tuple(self.dilation)
        self.rates[0] = torch.tensor(init_dilation, dtype=torch.int32)
        self.sample_weights.data.fill_(self.init_weight)

        if self.rf_mode == "rfsearch":
            self.estimate()
            self.expand()
        elif self.rf_mode == "rfsingle":
            self.estimate()
            self.max_search_step = 0
            self.sample_weights.requires_grad = False
        elif self.rf_mode == "rfmultiple":
            self.estimate()
            self.expand()
            self.sample_weights.data.fill_(self.init_weight)
            self.max_search_step = 0
        elif self.rf_mode == "rfmerge":
            self.max_search_step = 0
            self.sample_weights.requires_grad = False

        if self.rf_mode == "rfsingle" and self.num_rates.item() != 1:
            raise ValueError("rfsingle expects a single active receptive field.")

    def _conv_forward_dilation(self, input_tensor, dilation_rate):
        dilation_rate = _to_2tuple(dilation_rate)
        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "RFConv2d currently supports only zero padding mode."
            )

        padding = (
            dilation_rate[0] * (self.kernel_size[0] - 1) // 2,
            dilation_rate[1] * (self.kernel_size[1] - 1) // 2,
        )
        return F.conv2d(
            input_tensor,
            self.weight,
            self.bias,
            self.stride,
            padding,
            dilation_rate,
            self.groups,
        )

    def normalize(self, weights):
        abs_weights = torch.abs(weights)
        weight_sum = abs_weights.sum().clamp_min(1e-6)
        return abs_weights / weight_sum

    def _active_rates(self):
        num_rates = int(self.num_rates.item())
        if num_rates <= 0:
            return self.rates[:0].to(dtype=self.weight.dtype, device=self.weight.device)
        return self.rates[:num_rates].to(dtype=self.weight.dtype, device=self.weight.device)

    def tensor_to_tuple(self, tensor):
        return tuple((value[0].item(), value[1].item()) for value in tensor)

    def expected_dilation(self):
        active_rates = self._active_rates()
        if active_rates.numel() == 0:
            return self.weight.new_zeros(2)
        if active_rates.shape[0] == 1:
            return active_rates[0]
        norm_weights = self.normalize(self.sample_weights[: active_rates.shape[0]])
        return (active_rates * norm_weights[:, None].to(active_rates.dtype)).sum(dim=0)

    def schedule_state(self):
        stop_step = int(self.search_stop_step.item())
        return {
            "search_interval": int(self.search_interval),
            "max_search_step": int(self.max_search_step),
            "start_step": int(self.search_start_step.item()),
            "stop_step": None if stop_step < 0 else stop_step,
        }

    @torch.no_grad()
    def configure_search_schedule(
        self,
        schedule_mode="manual",
        steps_per_epoch=0,
        total_epochs=0,
        search_interval=None,
        max_search_step=None,
        warmup_epochs=0,
        search_epochs=0,
        reset_counters=True,
    ):
        if max_search_step is not None:
            self.max_search_step = max(0, int(max_search_step))

        derived_interval = (
            int(search_interval)
            if search_interval is not None
            else int(self.search_interval)
        )
        start_step = 0
        stop_step = -1

        if schedule_mode == "epoch":
            steps_per_epoch = max(1, int(steps_per_epoch))
            total_epochs = max(1, int(total_epochs))
            warmup_epochs = max(0, int(warmup_epochs))
            if search_epochs is None or int(search_epochs) <= 0:
                search_epochs = max(1, total_epochs - warmup_epochs)
            else:
                search_epochs = int(search_epochs)
            start_step = warmup_epochs * steps_per_epoch
            active_steps = max(1, search_epochs * steps_per_epoch)
            stop_step = start_step + active_steps
            if self.max_search_step > 0:
                derived_interval = max(1, active_steps // self.max_search_step)
            else:
                derived_interval = max(1, derived_interval)
        else:
            derived_interval = max(1, derived_interval)

        self.search_interval = int(derived_interval)
        self.search_start_step.fill_(int(start_step))
        self.search_stop_step.fill_(int(stop_step))

        if reset_counters:
            self.counter.zero_()
            self.forward_step.zero_()
            self.current_search_step.zero_()

        return self.schedule_state()

    @torch.no_grad()
    def estimate(self):
        norm_weights = self.normalize(self.sample_weights[: self.num_rates.item()])
        sum_h = 0.0
        sum_w = 0.0
        weight_sum = 0.0
        for index in range(self.num_rates.item()):
            sum_h += norm_weights[index].item() * self.rates[index][0].item()
            sum_w += norm_weights[index].item() * self.rates[index][1].item()
            weight_sum += norm_weights[index].item()

        estimated = (
            value_crop(
                int(round(sum_h / max(weight_sum, 1e-6))),
                self.min_dilation,
                self.max_dilation,
            ),
            value_crop(
                int(round(sum_w / max(weight_sum, 1e-6))),
                self.min_dilation,
                self.max_dilation,
            ),
        )
        self.dilation = estimated
        self.padding = (
            _same_padding(self.kernel_size[0], self.stride[0], self.dilation[0]),
            _same_padding(self.kernel_size[1], self.stride[1], self.dilation[1]),
        )
        self.rates[0] = torch.tensor(estimated, dtype=torch.int32, device=self.rates.device)
        self.num_rates[0] = 1

    @torch.no_grad()
    def expand(self):
        rates = rf_expand(
            self.dilation,
            self.expand_rate,
            self.num_branches,
            min_dilation=self.min_dilation,
            max_dilation=self.max_dilation,
        )
        for index, rate in enumerate(rates):
            self.rates[index] = torch.tensor(rate, dtype=torch.int32, device=self.rates.device)
        self.num_rates[0] = len(rates)
        self.sample_weights.data.fill_(self.init_weight)

    @torch.no_grad()
    def searcher(self):
        if self.max_search_step == 0 or bool(self.merged.item()):
            return

        self.forward_step += 1
        current_step = int(self.forward_step.item())
        if current_step <= int(self.search_start_step.item()):
            return

        stop_step = int(self.search_stop_step.item())
        if stop_step >= 0 and current_step > stop_step:
            return

        self.counter += 1
        if (
            self.counter % self.search_interval == 0
            and self.current_search_step < self.max_search_step
            and self.max_search_step != 0
        ):
            self.counter[0] = 0
            self.current_search_step += 1
            self.estimate()
            self.expand()

    @torch.no_grad()
    def merge_branches_(self):
        if bool(self.merged.item()):
            return self.rf_state()

        num_rates = int(self.num_rates.item())
        if num_rates <= 0:
            return self.rf_state()

        active_rates = self._active_rates()
        if active_rates.shape[0] == 1:
            norm_weights = self.weight.new_ones(1)
        else:
            norm_weights = self.normalize(self.sample_weights[: active_rates.shape[0]]).to(
                dtype=self.weight.dtype,
                device=self.weight.device,
            )

        old_weight = self.weight.detach()
        old_bias = None if self.bias is None else self.bias.detach().clone()
        old_kernel_h, old_kernel_w = self.kernel_size
        center_h = old_kernel_h // 2
        center_w = old_kernel_w // 2

        max_rate_h = max(int(rate[0].item()) for rate in active_rates)
        max_rate_w = max(int(rate[1].item()) for rate in active_rates)
        new_kernel_size = (
            old_kernel_h + (max_rate_h - 1) * center_h * 2,
            old_kernel_w + (max_rate_w - 1) * center_w * 2,
        )
        new_center_h = new_kernel_size[0] // 2
        new_center_w = new_kernel_size[1] // 2
        in_channels_per_group = old_weight.shape[1]
        merged_weight = old_weight.new_zeros(
            self.out_channels,
            in_channels_per_group,
            new_kernel_size[0],
            new_kernel_size[1],
        )

        for branch_index, branch_rate in enumerate(active_rates):
            rate_h = int(branch_rate[0].item())
            rate_w = int(branch_rate[1].item())
            branch_weight = norm_weights[branch_index]
            for old_h in range(old_kernel_h):
                for old_w in range(old_kernel_w):
                    offset_h = old_h - center_h
                    offset_w = old_w - center_w
                    new_h = new_center_h + offset_h * rate_h
                    new_w = new_center_w + offset_w * rate_w
                    merged_weight[:, :, new_h, new_w] += (
                        old_weight[:, :, old_h, old_w] * branch_weight
                    )

        self.weight = nn.Parameter(merged_weight)
        if old_bias is not None:
            self.bias = nn.Parameter(old_bias)
        self.kernel_size = new_kernel_size
        self.dilation = (1, 1)
        self.padding = (
            _same_padding(self.kernel_size[0], self.stride[0], 1),
            _same_padding(self.kernel_size[1], self.stride[1], 1),
        )
        self.rates.zero_()
        self.rates[0] = torch.tensor([1, 1], dtype=torch.int32, device=self.rates.device)
        self.num_rates[0] = 1
        self.sample_weights.data.fill_(1.0)
        self.sample_weights.requires_grad = False
        self.merged.fill_(True)
        self.rf_mode = "rfmerge"
        self.counter.zero_()
        self.forward_step.zero_()
        self.current_search_step.zero_()
        return self.rf_state()

    def rf_state(self):
        num_rates = int(self.num_rates.item())
        if num_rates <= 0:
            return {}
        if num_rates == 1:
            weights = [1.0]
        else:
            weights = (
                self.normalize(self.sample_weights[:num_rates])
                .detach()
                .cpu()
                .tolist()
            )
        return {
            "mode": self.rf_mode,
            "merged": bool(self.merged.item()),
            "kernel_size": tuple(int(value) for value in self.kernel_size),
            "dilation": tuple(int(value) for value in self.dilation),
            "rates": self.tensor_to_tuple(self.rates[:num_rates].detach().cpu()),
            "weights": weights,
            "search_step": int(self.current_search_step.item()),
            **self.schedule_state(),
        }

    def forward(self, x):
        if self.num_rates.item() == 1:
            return super().forward(x)

        norm_weights = self.normalize(self.sample_weights[: self.num_rates.item()])
        outputs = [
            self._conv_forward_dilation(
                x,
                (
                    self.rates[index][0].item(),
                    self.rates[index][1].item(),
                ),
            )
            * norm_weights[index]
            for index in range(self.num_rates.item())
        ]
        output = outputs[0]
        for index in range(1, self.num_rates.item()):
            output = output + outputs[index]

        if self.training and self.rf_mode == "rfsearch":
            self.searcher()
        return output


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
            group_norm(in_dim),
            act(inplace=True),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            group_norm(out_dim),
            act(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RFDepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        act=nn.SiLU,
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
        self.depthwise = RFConv2d(
            in_dim,
            in_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=max(1, int(dilation)),
            groups=in_dim,
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
        self.depth_bn = group_norm(in_dim)
        self.depth_act = act(inplace=True)
        self.pointwise = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        self.point_bn = group_norm(out_dim)
        self.point_act = act(inplace=True)

    def rf_state(self):
        return self.depthwise.rf_state()

    def expected_dilation(self):
        return self.depthwise.expected_dilation()

    def configure_search_schedule(self, **kwargs):
        return self.depthwise.configure_search_schedule(**kwargs)

    def merge_branches_(self):
        return self.depthwise.merge_branches_()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.depth_bn(x)
        x = self.depth_act(x)
        x = self.pointwise(x)
        x = self.point_bn(x)
        x = self.point_act(x)
        return x


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
    def __init__(
        self,
        dim: int,
        rates=(1, 2, 4, 8),
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
        if not rates:
            raise ValueError("ASPPContext expects at least one dilation rate.")

        rates = tuple(max(1, int(rate)) for rate in rates)
        self.rf_enable = bool(rf_enable)
        if rf_max_dilations is None:
            rf_max_dilations = [None] * len(rates)
        elif len(rf_max_dilations) != len(rates):
            raise ValueError(
                "ASPPContext expects rf_max_dilations to match the ASPP branch count."
            )

        branch_cls = RFDepthwiseSeparableConv if self.rf_enable else DepthwiseSeparableConv
        branch_kwargs = []
        for rate, rf_max_dilation in zip(rates, rf_max_dilations):
            if self.rf_enable:
                branch_kwargs.append(
                    dict(
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
            else:
                branch_kwargs.append({})

        self.branches = nn.ModuleList(
            [
                branch_cls(
                    dim,
                    dim,
                    kernel_size=3,
                    dilation=rate,
                    **kwargs,
                )
                for rate, kwargs in zip(rates, branch_kwargs)
            ]
        )
        self.project = ConvBnAct(dim * len(self.branches), dim, kernel_size=1)

    def rf_states(self):
        if not self.rf_enable:
            return []
        return [branch.rf_state() for branch in self.branches]

    def configure_rf_search(self, **kwargs):
        if not self.rf_enable:
            return None
        branch_summaries = [
            branch.configure_search_schedule(**kwargs) for branch in self.branches
        ]
        summary = branch_summaries[0].copy() if branch_summaries else {}
        summary["branches"] = branch_summaries
        return summary

    def merge_rf_branches_(self):
        if not self.rf_enable:
            return []
        return [branch.merge_branches_() for branch in self.branches]

    def rf_diversity_loss(self, margin=1.0):
        reference = self.project.block[0].weight
        if not self.rf_enable or len(self.branches) < 2:
            return reference.new_zeros(())

        margin = float(margin)
        expected = torch.stack(
            [branch.expected_dilation().mean() for branch in self.branches],
            dim=0,
        )
        loss = expected.new_zeros(())
        count = 0
        for left in range(len(self.branches) - 1):
            for right in range(left + 1, len(self.branches)):
                target_gap = margin * (right - left)
                loss = loss + F.relu(target_gap - (expected[right] - expected[left]))
                count += 1
        return loss / max(count, 1)

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
        self.depth_fuse = DepthAttnFuse(in_dim=in_dim, hidden_dim=mid_dim, num_layers=4)
        self.context = ASPPContext(
            mid_dim,
            rates=tuple(aspp_rates),
            rf_enable=rf_enable,
            rf_mode=rf_mode,
            rf_num_branches=rf_num_branches,
            rf_expand_rate=rf_expand_rate,
            rf_min_dilation=rf_min_dilation,
            rf_max_dilations=rf_max_dilations,
            rf_search_interval=rf_search_interval,
            rf_max_search_step=rf_max_search_step,
            rf_init_weight=rf_init_weight,
        )
        self.refine_blocks = nn.ModuleList(
            [DepthwiseSeparableConv(mid_dim, mid_dim, kernel_size=3) for _ in range(4)]
        )
        self.out_blocks = nn.ModuleList(
            [nn.Conv2d(mid_dim, out_dim, kernel_size=1, bias=True) for _ in range(4)]
        )
        self.scales = (2.0, 1.0, 0.5, 0.25)

    def rf_states(self):
        return self.context.rf_states()

    def configure_rf_search(self, **kwargs):
        return self.context.configure_rf_search(**kwargs)

    def merge_rf_branches_(self):
        return self.context.merge_rf_branches_()

    def rf_diversity_loss(self, margin=1.0):
        return self.context.rf_diversity_loss(margin=margin)

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
