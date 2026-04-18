import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .blocks.fpn import FPN, DsBnRelu
from .blocks.cbam import CBAM
from .blocks.adapter import DINOV3Wrapper, DenseAdapterLite
from .blocks.change_prior import AdaptiveChangePriorPyramid
from .blocks.diffatts import TransformerBlock
from .blocks.mfce import MFCEPyramidAdapter, RFConv2d, TemporalFeatureExchange
from .blocks.pair_local import PairLocalPyramid
from .blocks.refine import LearnableSoftMorph
from .backbone.mobilenetv2 import mobilenet_v2


def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


class PyramidFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims=[128, 128, 128, 128],
        dense_dim=1024,
        patch_size=16,
        hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.c4 = nn.Sequential(
            DsBnRelu(in_dims[3] + hidden_dim, in_dims[3]), CBAM(in_dims[3], 8)
        )
        self.c3 = nn.Sequential(
            DsBnRelu(in_dims[2] + hidden_dim, in_dims[2]), CBAM(in_dims[2], 8)
        )
        self.c2 = nn.Sequential(
            DsBnRelu(in_dims[1] + hidden_dim, in_dims[1]), CBAM(in_dims[1], 8)
        )
        self.c1 = nn.Sequential(
            DsBnRelu(in_dims[0] + hidden_dim, in_dims[0]), CBAM(in_dims[0], 8)
        )

    def forward(self, feas, ds_feas):
        # process backbone (CNN) features
        x1, x2, x3, x4 = (
            feas  # [B, 128, 64, 64], [B, 128, 32, 32], [B, 128, 16, 16], [B, 128, 8, 8]
        )
        a1, a2, a3, a4 = (
            ds_feas  # [B, 256, 64, 64], [B, 256, 32, 32], [B, 256, 16, 16], [B, 256, 8, 8]
        )

        x4 = torch.cat([x4, a4], 1)
        x4 = self.c4(x4)

        x3 = torch.cat([x3, a3], 1)
        x3 = self.c3(x3)

        x2 = torch.cat([x2, a2], 1)
        x2 = self.c2(x2)

        x1 = torch.cat([x1, a1], 1)
        x1 = self.c1(x1)

        return x1, x2, x3, x4


class RFConvBnAct(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=3,
        stride=1,
        act=nn.SiLU,
        rf_mode="rfsearch",
        rf_num_branches=3,
        rf_expand_rate=0.5,
        rf_min_dilation=1,
        rf_max_dilation=None,
        rf_search_interval=100,
        rf_max_search_step=8,
        rf_init_weight=0.01,
    ):
        super().__init__()
        self.conv = RFConv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=1,
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
        self.bn = nn.BatchNorm2d(dim)
        self.act = act(inplace=True)

    def rf_state(self):
        return self.conv.rf_state()

    def configure_search_schedule(self, **kwargs):
        return self.conv.configure_search_schedule(**kwargs)

    def merge_branches_(self):
        return self.conv.merge_branches_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        backbone="mobilenetv2",
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        beta_mode="contextgatedconv",
        dino_weight="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device="cuda",
        extract_ids=[5, 11, 17, 23],
        **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(
            weights_path=dino_weight,
            device=device,
            extract_ids=extract_ids,
            use_lora=kwargs.get("dino_lora", False),
            use_dora=kwargs.get("dino_dora", False),
            lora_r=kwargs.get("dino_lora_r", 8),
            lora_alpha=kwargs.get("dino_lora_alpha", 16),
            lora_dropout=kwargs.get("dino_lora_dropout", 0.05),
            lora_search=kwargs.get("dino_lora_search", False),
            lora_r_target=kwargs.get("dino_lora_r_target", 4),
            lora_alpha_over_r=kwargs.get("dino_lora_alpha_over_r", 1.0),
            lora_search_warmup_epochs=kwargs.get("dino_lora_search_warmup_epochs", 5),
            lora_search_interval=kwargs.get("dino_lora_search_interval", 1),
            lora_search_ema_decay=kwargs.get("dino_lora_search_ema_decay", 0.9),
            lora_search_score_norm=kwargs.get("dino_lora_search_score_norm", "median"),
            lora_search_grad_weight=kwargs.get("dino_lora_search_grad_weight", 0.5),
            lora_search_budget_mode=kwargs.get("dino_lora_search_budget_mode", "grouped"),
            lora_search_group_weights=kwargs.get("dino_lora_search_group_weights", None),
            lora_search_depth_buckets=kwargs.get("dino_lora_search_depth_buckets", 3),
            lora_search_counterfactual=kwargs.get("dino_lora_search_counterfactual", False),
            lora_search_counterfactual_val_batches=kwargs.get(
                "dino_lora_search_counterfactual_val_batches", 2
            ),
            lora_search_counterfactual_max_candidates=kwargs.get(
                "dino_lora_search_counterfactual_max_candidates", 64
            ),
            lora_search_counterfactual_delta=kwargs.get(
                "dino_lora_search_counterfactual_delta", 0.0
            ),
            lora_search_counterfactual_patience=kwargs.get(
                "dino_lora_search_counterfactual_patience", 1
            ),
        )
        if kwargs.get("mfce_enable", False):
            self.dense_adp = MFCEPyramidAdapter(
                in_dim=1024,
                out_dim=dense_out_dim,
                mid_dim=kwargs.get("mfce_mid_dim", dense_out_dim),
                aspp_rates=tuple(kwargs.get("mfce_aspp_rates", [1, 2, 4, 8])),
                rf_enable=kwargs.get("mfce_rf_enable", False),
                rf_mode=kwargs.get("mfce_rf_mode", "rfsearch"),
                rf_num_branches=kwargs.get("mfce_rf_num_branches", 3),
                rf_expand_rate=kwargs.get("mfce_rf_expand_rate", 0.5),
                rf_min_dilation=kwargs.get("mfce_rf_min_dilation", 1),
                rf_max_dilations=kwargs.get("mfce_rf_max_dilations", None),
                rf_search_interval=kwargs.get("mfce_rf_search_interval", 100),
                rf_max_search_step=kwargs.get("mfce_rf_max_search_step", 8),
                rf_init_weight=kwargs.get("mfce_rf_init_weight", 0.01),
            )
        else:
            self.dense_adp = DenseAdapterLite(
                in_dim=1024, out_dim=dense_out_dim, bottleneck=fpn_channels // 2
            )
        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )

    def forward_local(self, x):
        fea = self.backbone.forward(x)
        return self.fpn(fea[-4:])  # p2, p3, p4, p5

    def forward_dense_layers(self, x):
        return self.dino(x)

    def adapt_dense(self, dense_feats):
        return self.dense_adp(dense_feats)

    def forward_dense(self, x):
        ds_fea = self.forward_dense_layers(x)
        return self.adapt_dense(ds_fea)

    def fuse_pyramid(self, local_feas, ds_feas):
        return self.pff(local_feas, ds_feas)

    def forward(self, x):
        """
        x1: [B, 3, H, W]
        x2: [B, 3, H, W]
        return: [B, 1, H, W]
        """
        local_feas = self.forward_local(x)
        ds_fea = self.forward_dense(x)
        return self.fuse_pyramid(local_feas, ds_fea)


class FuseGated(nn.Module):
    def __init__(
        self,
        dim,
        rf_enable=False,
        rf_mode="rfsearch",
        rf_num_branches=3,
        rf_expand_rate=0.5,
        rf_min_dilation=1,
        rf_max_dilation=None,
        rf_search_interval=100,
        rf_max_search_step=8,
        rf_init_weight=0.01,
    ):
        super().__init__()
        self.rf_enable = bool(rf_enable)
        self.gate = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=True), nn.Sigmoid())
        if self.rf_enable:
            self.mix = RFConvBnAct(
                dim,
                kernel_size=3,
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
            self.mix = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU(inplace=True),
            )

    def rf_state(self):
        if not self.rf_enable:
            return {}
        return self.mix.rf_state()

    def configure_rf_search(self, **kwargs):
        if not self.rf_enable:
            return None
        return self.mix.configure_search_schedule(**kwargs)

    def merge_rf_branches_(self):
        if not self.rf_enable:
            return None
        return self.mix.merge_branches_()

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)


class Detector(nn.Module):
    def __init__(
        self,
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        **kwargs,
    ):
        super().__init__()
        self.acpc_enable = kwargs.get("acpc_enable", False)
        self.decoder_rf_enable = kwargs.get("decoder_rf_enable", False)
        self.change_prior = None
        if self.acpc_enable:
            self.change_prior = AdaptiveChangePriorPyramid(
                dim=fpn_channels,
                stage_modes=tuple(
                    kwargs.get("acpc_stage_modes", ["full", "lite", "none", "none"])
                ),
                hidden_ratio=kwargs.get("acpc_hidden_ratio", 0.5),
                norm_groups=kwargs.get("acpc_norm_groups", 8),
                residual_scale=kwargs.get("acpc_residual_scale", 0.05),
            )
        decoder_rf_max_dilations = kwargs.get("decoder_rf_max_dilations", None)
        if decoder_rf_max_dilations is None:
            decoder_rf_max_dilations = [None, None, None]
        self.p5_to_p4 = FuseGated(
            fpn_channels,
            rf_enable=self.decoder_rf_enable,
            rf_mode=kwargs.get("decoder_rf_mode", "rfsearch"),
            rf_num_branches=kwargs.get("decoder_rf_num_branches", 3),
            rf_expand_rate=kwargs.get("decoder_rf_expand_rate", 0.5),
            rf_min_dilation=kwargs.get("decoder_rf_min_dilation", 1),
            rf_max_dilation=decoder_rf_max_dilations[0],
            rf_search_interval=kwargs.get("decoder_rf_search_interval", 100),
            rf_max_search_step=kwargs.get("decoder_rf_max_search_step", 8),
            rf_init_weight=kwargs.get("decoder_rf_init_weight", 0.01),
        )
        self.p4_to_p3 = FuseGated(
            fpn_channels,
            rf_enable=self.decoder_rf_enable,
            rf_mode=kwargs.get("decoder_rf_mode", "rfsearch"),
            rf_num_branches=kwargs.get("decoder_rf_num_branches", 3),
            rf_expand_rate=kwargs.get("decoder_rf_expand_rate", 0.5),
            rf_min_dilation=kwargs.get("decoder_rf_min_dilation", 1),
            rf_max_dilation=decoder_rf_max_dilations[1],
            rf_search_interval=kwargs.get("decoder_rf_search_interval", 100),
            rf_max_search_step=kwargs.get("decoder_rf_max_search_step", 8),
            rf_init_weight=kwargs.get("decoder_rf_init_weight", 0.01),
        )
        self.p3_to_p2 = FuseGated(
            fpn_channels,
            rf_enable=self.decoder_rf_enable,
            rf_mode=kwargs.get("decoder_rf_mode", "rfsearch"),
            rf_num_branches=kwargs.get("decoder_rf_num_branches", 3),
            rf_expand_rate=kwargs.get("decoder_rf_expand_rate", 0.5),
            rf_min_dilation=kwargs.get("decoder_rf_min_dilation", 1),
            rf_max_dilation=decoder_rf_max_dilations[2],
            rf_search_interval=kwargs.get("decoder_rf_search_interval", 100),
            rf_max_search_step=kwargs.get("decoder_rf_max_search_step", 8),
            rf_init_weight=kwargs.get("decoder_rf_init_weight", 0.01),
        )

        self.tb5 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[0])
            ]
        )
        self.tb4 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[1])
            ]
        )
        self.tb3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=2,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[2])
            ]
        )
        self.tb2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=1,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[3])
            ]
        )
        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)

    def _iter_rf_fuse_modules(self):
        if not self.decoder_rf_enable:
            return []
        return [
            ("p5_to_p4", self.p5_to_p4),
            ("p4_to_p3", self.p4_to_p3),
            ("p3_to_p2", self.p3_to_p2),
        ]

    def rf_states(self):
        return [
            {"name": name, **module.rf_state()}
            for name, module in self._iter_rf_fuse_modules()
        ]

    def configure_rf_search(self, **kwargs):
        stage_summaries = []
        for name, module in self._iter_rf_fuse_modules():
            summary = module.configure_rf_search(**kwargs)
            if summary is not None:
                stage_summaries.append({"name": name, **summary})
        if not stage_summaries:
            return None
        summary = stage_summaries[0].copy()
        summary["stages"] = stage_summaries
        return summary

    def merge_rf_branches_(self):
        return [
            {"name": name, **module.merge_rf_branches_()}
            for name, module in self._iter_rf_fuse_modules()
        ]

    def _build_change_priors(self, x1s, x2s):
        if self.change_prior is not None:
            return self.change_prior(x1s, x2s)

        return tuple(torch.abs(feat1 - feat2) for feat1, feat2 in zip(x1s, x2s))

    def forward(self, x1s, x2s, size=(256, 256)):
        ### Extract backbone features
        diff_p2, diff_p3, diff_p4, diff_p5 = self._build_change_priors(x1s, x2s)

        fea_p5 = self.tb5(diff_p5)
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4)
        fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        fea_p3 = self.tb3(fea_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        fea_p2 = self.tb2(fea_p2)
        pred_p2 = self.p2_head(fea_p2)

        pred_p2 = F.interpolate(
            pred_p2, size=size, mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=size, mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=size, mode="bilinear", align_corners=False
        )
        pred_p5 = F.interpolate(
            pred_p5, size=size, mode="bilinear", align_corners=False
        )

        return pred_p2, pred_p3, pred_p4, pred_p5


class ChangeModel(nn.Module):
    def __init__(
        self, backbone="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1, 1], **kwargs
    ):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.pairlocal_enable = kwargs.get("pairlocal_enable", False)
        self.pairlocal = None
        if self.pairlocal_enable:
            self.pairlocal = PairLocalPyramid(
                dim=fpn_channels,
                stage_modes=tuple(
                    kwargs.get(
                        "pairlocal_stage_modes",
                        ["full", "full", "lite", "lite"],
                    )
                ),
                hidden_ratio=kwargs.get("pairlocal_hidden_ratio", 1.0),
                norm_groups=kwargs.get("pairlocal_norm_groups", 8),
                residual_scale=kwargs.get("pairlocal_residual_scale", 0.1),
                rf_enable=kwargs.get("pairlocal_rf_enable", False),
                rf_mode=kwargs.get("pairlocal_rf_mode", "rfsearch"),
                rf_num_branches=kwargs.get("pairlocal_rf_num_branches", 3),
                rf_expand_rate=kwargs.get("pairlocal_rf_expand_rate", 0.5),
                rf_min_dilation=kwargs.get("pairlocal_rf_min_dilation", 1),
                rf_max_dilations=kwargs.get("pairlocal_rf_max_dilations", None),
                rf_search_interval=kwargs.get("pairlocal_rf_search_interval", 100),
                rf_max_search_step=kwargs.get("pairlocal_rf_max_search_step", 8),
                rf_init_weight=kwargs.get("pairlocal_rf_init_weight", 0.01),
            )
        self.temporal_exchange = None
        if kwargs.get("dino_temporal_exchange_enable", False):
            self.temporal_exchange = TemporalFeatureExchange(
                mode=kwargs.get("dino_temporal_exchange_mode", "layer"),
                thresh=kwargs.get("dino_temporal_exchange_thresh", 0.5),
                p=kwargs.get("dino_temporal_exchange_p", 2),
                layers=tuple(kwargs.get("dino_temporal_exchange_layers", [0, 1, 2, 3])),
            )
        self.detector = Detector(fpn_channels=fpn_channels, n_layers=n_layers, **kwargs)
        self.refiner = LearnableSoftMorph(3, 5)

    def _encode_pair(self, x1, x2):
        local1 = self.encoder.forward_local(x1)
        local2 = self.encoder.forward_local(x2)
        if self.pairlocal_enable:
            local1, local2 = self.pairlocal(local1, local2)

        dense_layers1 = self.encoder.forward_dense_layers(x1)
        dense_layers2 = self.encoder.forward_dense_layers(x2)
        if self.temporal_exchange is not None:
            dense_layers1, dense_layers2 = self.temporal_exchange(
                dense_layers1,
                dense_layers2,
            )
        dense1 = self.encoder.adapt_dense(dense_layers1)
        dense2 = self.encoder.adapt_dense(dense_layers2)
        fea1 = self.encoder.fuse_pyramid(local1, dense1)
        fea2 = self.encoder.fuse_pyramid(local2, dense2)
        return fea1, fea2

    @torch.inference_mode()
    def _forward(self, x1, x2):
        # for inference
        fea1, fea2 = self._encode_pair(x1, x2)
        pred, _, _, _ = self.detector(fea1, fea2, x1.shape[-2:])
        pred = self.refiner(pred)
        return pred

    def forward(self, x1, x2):
        # for training
        ## change detection
        fea1, fea2 = self._encode_pair(x1, x2)

        preds = self.detector(fea1, fea2, x1.shape[-2:])
        final_pred = self.refiner(preds[0])
        return final_pred, preds  # pred, pred_p2, pred_p3, pred_p4, pred_p5
