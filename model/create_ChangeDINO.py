from .ChangeDINO import ChangeModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
import os
import torch.optim as optim
from .loss.focal import FocalLoss
from .loss.dice import DICELoss
from util.metric_tool import cm2score, get_confuse_matrix


def get_model(backbone_name="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1], **kwargs):
    model = ChangeModel(backbone_name, fpn_channels, n_layers=n_layers, **kwargs)
    # print(model)
    return model


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device(
            "cuda:%s" % opt.device_id if torch.cuda.is_available() else "cpu"
        )
        self.use_distributed = getattr(opt, "distributed", False)
        self.use_data_parallel = (
            not self.use_distributed
            and torch.cuda.is_available()
            and len(opt.gpu_ids) > 1
        )
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_aux_losses = {}
        self.last_spectral_probe_stats = {}
        self.last_spectral_search_stats = {}
        self.current_epoch = 0

        self.model = get_model(
            backbone_name=opt.backbone,
            fpn_name=opt.fpn,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            n_layers=opt.n_layers,
            extract_ids=opt.extract_ids,
            dino_local_conv_enable=opt.dino_local_conv_enable,
            dino_local_conv_blocks=opt.dino_local_conv_blocks,
            dino_local_conv_kernel_size=opt.dino_local_conv_kernel_size,
            dino_local_conv_init_scale=opt.dino_local_conv_init_scale,
            dino_local_conv_change_aware_enable=opt.dino_local_conv_change_aware_enable,
            dino_local_conv_change_hidden_ratio=opt.dino_local_conv_change_hidden_ratio,
            dino_local_conv_change_norm_groups=opt.dino_local_conv_change_norm_groups,
            dino_local_conv_change_residual_scale=opt.dino_local_conv_change_residual_scale,
            dino_local_conv_change_delta_scale=opt.dino_local_conv_change_delta_scale,
            dino_local_conv_change_mixer_enable=opt.dino_local_conv_change_mixer_enable,
            dino_local_conv_change_mixer_kernel_size=opt.dino_local_conv_change_mixer_kernel_size,
            dino_local_conv_change_mixer_residual_scale=opt.dino_local_conv_change_mixer_residual_scale,
            dino_local_conv_rf_enable=opt.dino_local_conv_rf_enable,
            dino_local_conv_rf_mode=opt.dino_local_conv_rf_mode,
            dino_local_conv_rf_num_branches=opt.dino_local_conv_rf_num_branches,
            dino_local_conv_rf_expand_rate=opt.dino_local_conv_rf_expand_rate,
            dino_local_conv_rf_min_dilation=opt.dino_local_conv_rf_min_dilation,
            dino_local_conv_rf_max_dilations=opt.dino_local_conv_rf_max_dilations,
            dino_local_conv_rf_search_interval=opt.dino_local_conv_rf_search_interval,
            dino_local_conv_rf_max_search_step=opt.dino_local_conv_rf_max_search_step,
            dino_local_conv_rf_init_weight=opt.dino_local_conv_rf_init_weight,
            dino_lora=opt.dino_lora,
            dino_dora=opt.dino_dora,
            dino_lora_r=opt.dino_lora_r,
            dino_lora_alpha=opt.dino_lora_alpha,
            dino_lora_dropout=opt.dino_lora_dropout,
            dino_lora_search=opt.dino_lora_search,
            dino_lora_soft_gate=opt.dino_lora_soft_gate,
            dino_lora_soft_gate_init=opt.dino_lora_soft_gate_init,
            dino_lora_soft_gate_temperature=opt.dino_lora_soft_gate_temperature,
            dino_lora_r_target=opt.dino_lora_r_target,
            dino_lora_alpha_over_r=opt.dino_lora_alpha_over_r,
            dino_lora_search_warmup_epochs=opt.dino_lora_search_warmup_epochs,
            dino_lora_search_interval=opt.dino_lora_search_interval,
            dino_lora_search_ema_decay=opt.dino_lora_search_ema_decay,
            dino_lora_search_score_norm=opt.dino_lora_search_score_norm,
            dino_lora_search_grad_weight=opt.dino_lora_search_grad_weight,
            dino_lora_search_budget_mode=opt.dino_lora_search_budget_mode,
            dino_lora_search_group_weights=opt.dino_lora_search_group_weights,
            dino_lora_search_depth_buckets=opt.dino_lora_search_depth_buckets,
            dino_lora_search_strategy=opt.dino_lora_search_strategy,
            dino_lora_search_probe_batches=opt.dino_lora_search_probe_batches,
            dino_lora_search_probe_refresh_interval=opt.dino_lora_search_probe_refresh_interval,
            dino_lora_search_probe_score_norm=opt.dino_lora_search_probe_score_norm,
            dino_lora_search_probe_keep_ratio=opt.dino_lora_search_probe_keep_ratio,
            dino_lora_search_probe_module_keep_ratio=opt.dino_lora_search_probe_module_keep_ratio,
            dino_lora_search_rf_delta=opt.dino_lora_search_rf_delta,
            dino_lora_search_rf_temperature=opt.dino_lora_search_rf_temperature,
            dino_lora_search_counterfactual=opt.dino_lora_search_counterfactual,
            dino_lora_search_counterfactual_val_batches=opt.dino_lora_search_counterfactual_val_batches,
            dino_lora_search_counterfactual_max_candidates=opt.dino_lora_search_counterfactual_max_candidates,
            dino_lora_search_counterfactual_delta=opt.dino_lora_search_counterfactual_delta,
            dino_lora_search_counterfactual_patience=opt.dino_lora_search_counterfactual_patience,
            dino_lora_search_spectral=opt.dino_lora_search_spectral,
            dino_lora_spectral_prior_power=opt.dino_lora_spectral_prior_power,
            dino_lora_spectral_uncertainty_weight=opt.dino_lora_spectral_uncertainty_weight,
            dino_lora_spectral_init_scale=opt.dino_lora_spectral_init_scale,
            mfce_enable=opt.mfce_enable,
            mfce_mid_dim=opt.mfce_mid_dim,
            mfce_aspp_rates=opt.mfce_aspp_rates,
            mfce_rf_enable=opt.mfce_rf_enable,
            mfce_rf_mode=opt.mfce_rf_mode,
            mfce_rf_num_branches=opt.mfce_rf_num_branches,
            mfce_rf_expand_rate=opt.mfce_rf_expand_rate,
            mfce_rf_min_dilation=opt.mfce_rf_min_dilation,
            mfce_rf_max_dilations=opt.mfce_rf_max_dilations,
            mfce_rf_search_interval=opt.mfce_rf_search_interval,
            mfce_rf_max_search_step=opt.mfce_rf_max_search_step,
            mfce_rf_init_weight=opt.mfce_rf_init_weight,
            decoder_rf_enable=opt.decoder_rf_enable,
            decoder_rf_mode=opt.decoder_rf_mode,
            decoder_rf_num_branches=opt.decoder_rf_num_branches,
            decoder_rf_expand_rate=opt.decoder_rf_expand_rate,
            decoder_rf_min_dilation=opt.decoder_rf_min_dilation,
            decoder_rf_max_dilations=opt.decoder_rf_max_dilations,
            decoder_rf_search_interval=opt.decoder_rf_search_interval,
            decoder_rf_max_search_step=opt.decoder_rf_max_search_step,
            decoder_rf_init_weight=opt.decoder_rf_init_weight,
            decoder_pred_guided_enable=opt.decoder_pred_guided_enable,
            decoder_pred_guided_mode=opt.decoder_pred_guided_mode,
            decoder_cgla_prior_enable=opt.decoder_cgla_prior_enable,
            decoder_cgla_prior_mode=opt.decoder_cgla_prior_mode,
            decoder_cgla_prior_source=opt.decoder_cgla_prior_source,
            decoder_cgla_prior_train_mode=opt.decoder_cgla_prior_train_mode,
            decoder_cgla_prior_scale_init=opt.decoder_cgla_prior_scale_init,
            decoder_bifpn_enable=opt.decoder_bifpn_enable,
            decoder_bifpn_repeats=opt.decoder_bifpn_repeats,
            decoder_bifpn_eps=opt.decoder_bifpn_eps,
            decoder_cgla_bifpn_enable=opt.decoder_cgla_bifpn_enable,
            decoder_cgla_bifpn_prior_mode=opt.decoder_cgla_bifpn_prior_mode,
            decoder_cgla_bifpn_prior_train_mode=opt.decoder_cgla_bifpn_prior_train_mode,
            decoder_cgla_bifpn_prior_scale_init=opt.decoder_cgla_bifpn_prior_scale_init,
            decoder_cgla_bifpn_prior_bias_limit=opt.decoder_cgla_bifpn_prior_bias_limit,
            dino_temporal_exchange_enable=opt.dino_temporal_exchange_enable,
            dino_temporal_exchange_mode=opt.dino_temporal_exchange_mode,
            dino_temporal_exchange_thresh=opt.dino_temporal_exchange_thresh,
            dino_temporal_exchange_p=opt.dino_temporal_exchange_p,
            dino_temporal_exchange_layers=opt.dino_temporal_exchange_layers,
            pairlocal_enable=opt.pairlocal_enable,
            pairlocal_stage_modes=opt.pairlocal_stage_modes,
            pairlocal_hidden_ratio=opt.pairlocal_hidden_ratio,
            pairlocal_norm_groups=opt.pairlocal_norm_groups,
            pairlocal_residual_scale=opt.pairlocal_residual_scale,
            pairlocal_rf_enable=opt.pairlocal_rf_enable,
            pairlocal_rf_mode=opt.pairlocal_rf_mode,
            pairlocal_rf_num_branches=opt.pairlocal_rf_num_branches,
            pairlocal_rf_expand_rate=opt.pairlocal_rf_expand_rate,
            pairlocal_rf_min_dilation=opt.pairlocal_rf_min_dilation,
            pairlocal_rf_max_dilations=opt.pairlocal_rf_max_dilations,
            pairlocal_rf_search_interval=opt.pairlocal_rf_search_interval,
            pairlocal_rf_max_search_step=opt.pairlocal_rf_max_search_step,
            pairlocal_rf_init_weight=opt.pairlocal_rf_init_weight,
            acpc_enable=opt.acpc_enable,
            acpc_stage_modes=opt.acpc_stage_modes,
            acpc_hidden_ratio=opt.acpc_hidden_ratio,
            acpc_norm_groups=opt.acpc_norm_groups,
            acpc_residual_scale=opt.acpc_residual_scale,
        )
        self._log_trainable_parameters()
        self._log_spectral_search_state("init")
        if opt.is_main_process:
            print("[CGLA-BiFPN]")
            print(f"decoder_bifpn_enable = {opt.decoder_bifpn_enable}")
            print(f"decoder_cgla_bifpn_enable = {opt.decoder_cgla_bifpn_enable}")
            print(f"decoder_bifpn_repeats = {opt.decoder_bifpn_repeats}")
            print(
                f"decoder_cgla_bifpn_prior_mode = {opt.decoder_cgla_bifpn_prior_mode}"
            )
            print(
                "decoder_cgla_bifpn_prior_train_mode = "
                f"{opt.decoder_cgla_bifpn_prior_train_mode}"
            )
            print(
                "decoder_cgla_bifpn_prior_scale_init = "
                f"{opt.decoder_cgla_bifpn_prior_scale_init}"
            )
        if opt.load_pretrain:
            self.load_ckpt(self.model, None, opt.name, opt.backbone)
        should_merge_rf = opt.load_pretrain and (
            (
                getattr(opt, "dino_local_conv_rf_enable", False)
                and (
                    getattr(opt, "dino_local_conv_rf_merge_on_eval", False)
                    or getattr(opt, "dino_local_conv_rf_mode", "") == "rfmerge"
                )
            )
            or (
                getattr(opt, "mfce_rf_enable", False)
                and (
                    getattr(opt, "mfce_rf_merge_on_eval", False)
                    or getattr(opt, "mfce_rf_mode", "") == "rfmerge"
                )
            )
            or (
                getattr(opt, "decoder_rf_enable", False)
                and (
                    getattr(opt, "decoder_rf_merge_on_eval", False)
                    or getattr(opt, "decoder_rf_mode", "") == "rfmerge"
                )
            )
            or (
                getattr(opt, "pairlocal_rf_enable", False)
                and (
                    getattr(opt, "pairlocal_rf_merge_on_eval", False)
                    or getattr(opt, "pairlocal_rf_mode", "") == "rfmerge"
                )
            )
        )
        if should_merge_rf:
            self.merge_rf_branches()
            self._log_rf_states("merged")
        else:
            self._log_rf_states("initial")

        self.model = self.model.to(self.device)
        if self.use_distributed:
            self.model = DDP(
                self.model,
                device_ids=[opt.device_id],
                output_device=opt.device_id,
                find_unused_parameters=False,
            )
            if opt.is_main_process:
                print(
                    f"using DistributedDataParallel on rank {opt.rank}/{opt.world_size}"
                )
        elif self.use_data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
            print(f"using DataParallel on GPUs: {opt.gpu_ids}")

        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = DICELoss()

        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )
        self.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, opt.num_epochs, eta_min=1e-7
        )

        print("---------- Networks initialized -------------")

    @staticmethod
    def _unwrap_model(network):
        return network.module if isinstance(network, (nn.DataParallel, DDP)) else network

    def _log_trainable_parameters(self):
        trainable = [
            (name, param.numel())
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        total_trainable = sum(numel for _, numel in trainable)
        print(f"trainable parameter tensors: {len(trainable)}")
        print(f"trainable parameters total: {total_trainable}")

        if self.opt.dino_lora or getattr(self.opt, "dino_dora", False):
            dino_trainable = [
                (name, numel)
                for name, numel in trainable
                if name.startswith("encoder.dino.model.") or name.startswith("module.encoder.dino.model.")
            ]
            lora_trainable = [
                (name, numel)
                for name, numel in trainable
                if ".lora_" in name
                or ".dora_" in name
                or ".spectral_" in name
                or ".gate_logits" in name
            ]
            print(f"trainable DINO tensors: {len(dino_trainable)}")
            print(f"trainable DINO parameters total: {sum(numel for _, numel in dino_trainable)}")
            adapter_label = (
                "DoRA"
                if getattr(self.opt, "dino_dora", False)
                else (
                    "SoftGateLoRA"
                    if getattr(self.opt, "dino_lora_soft_gate", False)
                    else (
                        "SpectralLoRA"
                        if getattr(self.opt, "dino_lora_search_spectral", False)
                        else "LoRA"
                    )
                )
            )
            print(f"trainable {adapter_label} tensors: {len(lora_trainable)}")
            print(f"trainable {adapter_label} parameters total: {sum(numel for _, numel in lora_trainable)}")

            if not lora_trainable:
                raise RuntimeError(
                    f"{adapter_label} is enabled, but no trainable adapter parameters were found."
                )

            preview_limit = 12
            print(f"sample trainable {adapter_label} parameter names:")
            for name, _ in lora_trainable[:preview_limit]:
                print(f"  {name}")

    def _collect_dense_adapter(self):
        network = self._unwrap_model(self.model)
        encoder = getattr(network, "encoder", None)
        return getattr(encoder, "dense_adp", None)

    def _collect_detector(self):
        network = self._unwrap_model(self.model)
        return getattr(network, "detector", None)

    def _collect_detector_debug_state(self):
        detector = self._collect_detector()
        if detector is None or not hasattr(detector, "bifpn_debug_state"):
            return None
        return detector.bifpn_debug_state()

    def _collect_pairlocal(self):
        network = self._unwrap_model(self.model)
        return getattr(network, "pairlocal", None)

    def _collect_spectral_search_layers(self):
        network = self._unwrap_model(self.model)
        dino = getattr(getattr(network, "encoder", None), "dino", None)
        if dino is None or not hasattr(dino, "iter_searchable_lora_layers"):
            return []
        return [
            layer
            for layer in dino.iter_searchable_lora_layers()
            if hasattr(layer, "spectral_scale") and hasattr(layer, "debug_state")
        ]

    def _collect_soft_gate_debug_state(self):
        network = self._unwrap_model(self.model)
        dino = getattr(getattr(network, "encoder", None), "dino", None)
        if dino is None or not hasattr(dino, "soft_gate_debug_state"):
            return None
        return dino.soft_gate_debug_state()

    def _collect_cgla_priors(self):
        network = self._unwrap_model(self.model)
        encoder = getattr(network, "encoder", None)
        if encoder is None or not hasattr(encoder, "get_last_cgla_priors"):
            return []
        priors = encoder.get_last_cgla_priors()
        return priors if isinstance(priors, list) else []

    def _soft_gate_budget_lambda(self, epoch):
        if not getattr(self.opt, "dino_lora_soft_gate", False):
            return 0.0
        warmup_epochs = int(getattr(self.opt, "dino_lora_soft_gate_budget_warmup_epochs", 0))
        ramp_epochs = int(max(1, getattr(self.opt, "dino_lora_soft_gate_budget_ramp_epochs", 1)))
        if epoch < warmup_epochs:
            return 0.0
        progress = min((epoch - warmup_epochs) / max(1, ramp_epochs), 1.0)
        return float(getattr(self.opt, "dino_lora_soft_gate_budget_weight", 0.0)) * progress

    def _cgla_temporal_reg_lambda(self, epoch):
        if not getattr(self.opt, "cgla_temporal_reg_enable", False):
            return 0.0
        warmup_epochs = int(
            getattr(self.opt, "cgla_temporal_reg_warmup_epochs", 0)
        )
        ramp_epochs = int(
            max(1, getattr(self.opt, "cgla_temporal_reg_ramp_epochs", 1))
        )
        if epoch < warmup_epochs:
            return 0.0
        progress = min((epoch - warmup_epochs) / max(1, ramp_epochs), 1.0)
        return float(getattr(self.opt, "cgla_temporal_reg_weight", 0.0)) * progress

    def _resize_change_mask_for_cgla_reg(self, target, target_size):
        mode = getattr(self.opt, "cgla_temporal_reg_mask_downsample", "area")
        if mode == "nearest":
            return F.interpolate(target, size=target_size, mode="nearest").clamp(0.0, 1.0)
        if mode == "area":
            return F.interpolate(target, size=target_size, mode="area").clamp(0.0, 1.0)
        if mode == "occupancy":
            return F.adaptive_max_pool2d(target, output_size=target_size).clamp(0.0, 1.0)
        raise ValueError(
            f"Unsupported --cgla_temporal_reg_mask_downsample: {mode}"
        )

    def _cgla_temporal_regularization(self, target):
        debug = {
            "cgla_temporal_reg_layers": 0.0,
            "cgla_temporal_reg_response_mean": 0.0,
            "cgla_temporal_reg_change_loss": 0.0,
            "cgla_temporal_reg_unchange_loss": 0.0,
            "cgla_temporal_reg_mask_mean": 0.0,
            "cgla_temporal_reg_mask_nonzero_ratio": 0.0,
        }
        if not getattr(self.opt, "cgla_temporal_reg_enable", False):
            return None, debug

        priors = self._collect_cgla_priors()
        if not priors:
            return None, debug

        source = getattr(self.opt, "cgla_temporal_reg_source", "delta")
        margin = float(getattr(self.opt, "cgla_temporal_reg_margin", 0.1))
        change_weight = float(
            getattr(self.opt, "cgla_temporal_reg_change_weight", 1.0)
        )
        unchange_weight = float(
            getattr(self.opt, "cgla_temporal_reg_unchange_weight", 1.0)
        )
        detach_response = bool(
            getattr(self.opt, "cgla_temporal_reg_detach_response", False)
        )

        layer_losses = []
        response_means = []
        change_losses = []
        unchange_losses = []
        mask_means = []
        mask_nonzero_ratios = []

        target = target.float().unsqueeze(1)
        for prior in priors:
            response = prior.get(source)
            if response is None:
                continue
            if detach_response:
                response = response.detach()
            response = response.float().abs()
            mask = self._resize_change_mask_for_cgla_reg(
                target,
                target_size=response.shape[-2:],
            )
            change_mask = mask
            unchange_mask = 1.0 - mask

            change_count = change_mask.sum().clamp_min(1.0)
            unchange_count = unchange_mask.sum().clamp_min(1.0)

            unchange_loss = ((response ** 2) * unchange_mask).sum() / unchange_count
            change_loss = (
                (F.relu(margin - response) ** 2) * change_mask
            ).sum() / change_count
            layer_loss = (
                change_weight * change_loss + unchange_weight * unchange_loss
            )

            layer_losses.append(layer_loss)
            response_means.append(response.mean())
            change_losses.append(change_loss)
            unchange_losses.append(unchange_loss)
            mask_means.append(mask.mean())
            mask_nonzero_ratios.append(mask.gt(0).float().mean())

        if not layer_losses:
            return None, debug

        reg_loss = torch.stack(layer_losses).mean()
        debug = {
            "cgla_temporal_reg_layers": float(len(layer_losses)),
            "cgla_temporal_reg_response_mean": float(
                torch.stack(response_means).mean().detach().item()
            ),
            "cgla_temporal_reg_change_loss": float(
                torch.stack(change_losses).mean().detach().item()
            ),
            "cgla_temporal_reg_unchange_loss": float(
                torch.stack(unchange_losses).mean().detach().item()
            ),
            "cgla_temporal_reg_mask_mean": float(
                torch.stack(mask_means).mean().detach().item()
            ),
            "cgla_temporal_reg_mask_nonzero_ratio": float(
                torch.stack(mask_nonzero_ratios).mean().detach().item()
            ),
        }
        return reg_loss, debug

    def _summarize_spectral_search_layers(self):
        layers = self._collect_spectral_search_layers()
        if not layers:
            return None

        records = []
        for layer in layers:
            state = layer.debug_state()
            state["module_name"] = getattr(layer, "module_name", "unknown")
            records.append(state)

        init_worst = max(records, key=lambda item: item["init_equiv_error"])
        corr_worst = min(records, key=lambda item: item["prior_importance_rank_corr"])
        uncertainty_worst = max(records, key=lambda item: item["uncertainty_ratio_max"])
        return {
            "layer_count": len(records),
            "init_equiv_error_max": float(init_worst["init_equiv_error"]),
            "init_equiv_error_module": init_worst["module_name"],
            "scale_abs_median_mean": float(
                sum(item["scale_abs_median"] for item in records) / max(1, len(records))
            ),
            "scale_abs_max": float(max(item["scale_abs_max"] for item in records)),
            "prior_importance_rank_corr_mean": float(
                sum(item["prior_importance_rank_corr"] for item in records)
                / max(1, len(records))
            ),
            "prior_importance_rank_corr_min": float(
                corr_worst["prior_importance_rank_corr"]
            ),
            "prior_importance_rank_corr_min_module": corr_worst["module_name"],
            "uncertainty_ratio_median_mean": float(
                sum(item["uncertainty_ratio_median"] for item in records)
                / max(1, len(records))
            ),
            "uncertainty_ratio_max": float(uncertainty_worst["uncertainty_ratio_max"]),
            "uncertainty_ratio_max_module": uncertainty_worst["module_name"],
        }

    def _log_spectral_search_state(self, prefix="current"):
        if not getattr(self.opt, "dino_lora_search_spectral", False):
            return
        if not getattr(self.opt, "is_main_process", True):
            return
        summary = self._summarize_spectral_search_layers()
        if summary is None:
            return
        self.last_spectral_search_stats = {"prefix": prefix, **summary}
        print(
            f"Spectral search [{prefix}] -> layers={summary['layer_count']}, "
            f"init_err_max={summary['init_equiv_error_max']:.3e}"
            f" ({summary['init_equiv_error_module']}), "
            f"scale_med_mean={summary['scale_abs_median_mean']:.3e}, "
            f"scale_max={summary['scale_abs_max']:.3e}, "
            f"prior_corr_mean={summary['prior_importance_rank_corr_mean']:.3f}, "
            f"prior_corr_min={summary['prior_importance_rank_corr_min']:.3f}"
            f" ({summary['prior_importance_rank_corr_min_module']}), "
            f"unc_ratio_med_mean={summary['uncertainty_ratio_median_mean']:.3f}, "
            f"unc_ratio_max={summary['uncertainty_ratio_max']:.3f}"
            f" ({summary['uncertainty_ratio_max_module']})"
        )

    def _log_rf_states(self, prefix="current"):
        network = self._unwrap_model(self.model)
        dino = getattr(getattr(network, "encoder", None), "dino", None)
        if getattr(self.opt, "dino_local_conv_rf_enable", False) and dino is not None:
            states = dino.local_conv_rf_states() if hasattr(dino, "local_conv_rf_states") else []
            if states:
                print(f"{prefix} DINO local-conv RF states:")
                for state in states:
                    nested_branch_keys = [
                        key
                        for key in ("local_branch", "change_context")
                        if isinstance(state.get(key), dict)
                    ]
                    if nested_branch_keys:
                        print(f"  {state.get('name')}:")
                        for branch_key in nested_branch_keys:
                            branch_state = state.get(branch_key, {})
                            rates = ", ".join(
                                f"({rate[0]},{rate[1]})"
                                for rate in branch_state.get("rates", [])
                            )
                            weights = ", ".join(
                                f"{weight:.3f}"
                                for weight in branch_state.get("weights", [])
                            )
                            print(
                                f"    {branch_key}: mode={branch_state.get('mode')} "
                                f"dilation={branch_state.get('dilation')} kernel={branch_state.get('kernel_size')} "
                                f"rates=[{rates}] weights=[{weights}] merged={branch_state.get('merged', False)} "
                                f"search_step={branch_state.get('search_step', 0)} "
                                f"interval={branch_state.get('search_interval')} "
                                f"window=[{branch_state.get('start_step')},{branch_state.get('stop_step')}]"
                            )
                        continue
                    rates = ", ".join(
                        f"({rate[0]},{rate[1]})" for rate in state.get("rates", [])
                    )
                    weights = ", ".join(
                        f"{weight:.3f}" for weight in state.get("weights", [])
                    )
                    print(
                        f"  {state.get('name')}: mode={state.get('mode')} "
                        f"dilation={state.get('dilation')} kernel={state.get('kernel_size')} "
                        f"rates=[{rates}] weights=[{weights}] merged={state.get('merged', False)} "
                        f"search_step={state.get('search_step', 0)} "
                        f"interval={state.get('search_interval')} "
                        f"window=[{state.get('start_step')},{state.get('stop_step')}]"
                    )

        dense_adapter = self._collect_dense_adapter()
        if getattr(self.opt, "mfce_rf_enable", False) and dense_adapter is not None:
            states = dense_adapter.rf_states() if hasattr(dense_adapter, "rf_states") else []
            if states:
                print(f"{prefix} MFCE RF states:")
                for branch_index, state in enumerate(states):
                    rates = ", ".join(
                        f"({rate[0]},{rate[1]})" for rate in state.get("rates", [])
                    )
                    weights = ", ".join(
                        f"{weight:.3f}" for weight in state.get("weights", [])
                    )
                    print(
                        f"  branch{branch_index}: mode={state.get('mode')} "
                        f"dilation={state.get('dilation')} kernel={state.get('kernel_size')} "
                        f"rates=[{rates}] weights=[{weights}] merged={state.get('merged', False)} "
                        f"search_step={state.get('search_step', 0)} "
                        f"interval={state.get('search_interval')} "
                        f"window=[{state.get('start_step')},{state.get('stop_step')}]"
                    )

        detector = self._collect_detector()
        if getattr(self.opt, "decoder_rf_enable", False) and detector is not None:
            states = detector.rf_states() if hasattr(detector, "rf_states") else []
            if states:
                print(f"{prefix} Decoder RF states:")
                for state in states:
                    rates = ", ".join(
                        f"({rate[0]},{rate[1]})" for rate in state.get("rates", [])
                    )
                    weights = ", ".join(
                        f"{weight:.3f}" for weight in state.get("weights", [])
                    )
                    print(
                        f"  {state.get('name')}: mode={state.get('mode')} "
                        f"dilation={state.get('dilation')} kernel={state.get('kernel_size')} "
                        f"rates=[{rates}] weights=[{weights}] merged={state.get('merged', False)} "
                        f"search_step={state.get('search_step', 0)} "
                        f"interval={state.get('search_interval')} "
                        f"window=[{state.get('start_step')},{state.get('stop_step')}]"
                    )

        pairlocal = self._collect_pairlocal()
        if getattr(self.opt, "pairlocal_rf_enable", False) and pairlocal is not None:
            states = pairlocal.rf_states() if hasattr(pairlocal, "rf_states") else []
            if states:
                print(f"{prefix} PairLocal RF states:")
                for state in states:
                    rates = ", ".join(
                        f"({rate[0]},{rate[1]})" for rate in state.get("rates", [])
                    )
                    weights = ", ".join(
                        f"{weight:.3f}" for weight in state.get("weights", [])
                    )
                    print(
                        f"  {state.get('name')}: mode={state.get('mode')} "
                        f"dilation={state.get('dilation')} kernel={state.get('kernel_size')} "
                        f"rates=[{rates}] weights=[{weights}] merged={state.get('merged', False)} "
                        f"search_step={state.get('search_step', 0)} "
                        f"interval={state.get('search_interval')} "
                        f"window=[{state.get('start_step')},{state.get('stop_step')}]"
                    )

    def configure_rf_search(self, steps_per_epoch: int):
        network = self._unwrap_model(self.model)
        dino = getattr(getattr(network, "encoder", None), "dino", None)
        dense_adapter = self._collect_dense_adapter()
        detector = self._collect_detector()
        pairlocal = self._collect_pairlocal()
        summaries = {}

        if (
            getattr(self.opt, "dino_local_conv_rf_enable", False)
            and dino is not None
            and hasattr(dino, "configure_local_conv_rf_search")
        ):
            schedule_mode = getattr(self.opt, "dino_local_conv_rf_schedule_mode", "manual")
            summary = dino.configure_local_conv_rf_search(
                schedule_mode=schedule_mode,
                steps_per_epoch=max(1, int(steps_per_epoch)),
                total_epochs=self.opt.num_epochs,
                search_interval=self.opt.dino_local_conv_rf_search_interval,
                max_search_step=self.opt.dino_local_conv_rf_max_search_step,
                warmup_epochs=self.opt.dino_local_conv_rf_search_warmup_epochs,
                search_epochs=self.opt.dino_local_conv_rf_search_epochs,
            )
            if summary:
                summaries["dino_local_conv"] = summary
                if self.opt.is_main_process:
                    print(
                        "DINO local-conv RF search schedule -> "
                        f"mode={schedule_mode}, interval={summary.get('search_interval')}, "
                        f"max_steps={summary.get('max_search_step')}, "
                        f"window=[{summary.get('start_step')},{summary.get('stop_step')}]"
                    )

        if (
            getattr(self.opt, "mfce_rf_enable", False)
            and dense_adapter is not None
            and hasattr(dense_adapter, "configure_rf_search")
        ):
            schedule_mode = getattr(self.opt, "mfce_rf_schedule_mode", "manual")
            summary = dense_adapter.configure_rf_search(
                schedule_mode=schedule_mode,
                steps_per_epoch=max(1, int(steps_per_epoch)),
                total_epochs=self.opt.num_epochs,
                search_interval=self.opt.mfce_rf_search_interval,
                max_search_step=self.opt.mfce_rf_max_search_step,
                warmup_epochs=self.opt.mfce_rf_search_warmup_epochs,
                search_epochs=self.opt.mfce_rf_search_epochs,
            )
            if summary:
                summaries["mfce"] = summary
                if self.opt.is_main_process:
                    print(
                        "MFCE RF search schedule -> "
                        f"mode={schedule_mode}, interval={summary.get('search_interval')}, "
                        f"max_steps={summary.get('max_search_step')}, "
                        f"window=[{summary.get('start_step')},{summary.get('stop_step')}]"
                    )

        if (
            getattr(self.opt, "decoder_rf_enable", False)
            and detector is not None
            and hasattr(detector, "configure_rf_search")
        ):
            schedule_mode = getattr(self.opt, "decoder_rf_schedule_mode", "manual")
            summary = detector.configure_rf_search(
                schedule_mode=schedule_mode,
                steps_per_epoch=max(1, int(steps_per_epoch)),
                total_epochs=self.opt.num_epochs,
                search_interval=self.opt.decoder_rf_search_interval,
                max_search_step=self.opt.decoder_rf_max_search_step,
                warmup_epochs=self.opt.decoder_rf_search_warmup_epochs,
                search_epochs=self.opt.decoder_rf_search_epochs,
            )
            if summary:
                summaries["decoder"] = summary
                if self.opt.is_main_process:
                    print(
                        "Decoder RF search schedule -> "
                        f"mode={schedule_mode}, interval={summary.get('search_interval')}, "
                        f"max_steps={summary.get('max_search_step')}, "
                        f"window=[{summary.get('start_step')},{summary.get('stop_step')}]"
                    )
        if (
            getattr(self.opt, "pairlocal_rf_enable", False)
            and pairlocal is not None
            and hasattr(pairlocal, "configure_rf_search")
        ):
            schedule_mode = getattr(self.opt, "pairlocal_rf_schedule_mode", "manual")
            summary = pairlocal.configure_rf_search(
                schedule_mode=schedule_mode,
                steps_per_epoch=max(1, int(steps_per_epoch)),
                total_epochs=self.opt.num_epochs,
                search_interval=self.opt.pairlocal_rf_search_interval,
                max_search_step=self.opt.pairlocal_rf_max_search_step,
                warmup_epochs=self.opt.pairlocal_rf_search_warmup_epochs,
                search_epochs=self.opt.pairlocal_rf_search_epochs,
            )
            if summary:
                summaries["pairlocal"] = summary
                if self.opt.is_main_process:
                    print(
                        "PairLocal RF search schedule -> "
                        f"mode={schedule_mode}, interval={summary.get('search_interval')}, "
                        f"max_steps={summary.get('max_search_step')}, "
                        f"window=[{summary.get('start_step')},{summary.get('stop_step')}]"
                    )
        return summaries if summaries else None

    def merge_rf_branches(self):
        merged = {}
        network = self._unwrap_model(self.model)
        dino = getattr(getattr(network, "encoder", None), "dino", None)
        if (
            getattr(self.opt, "dino_local_conv_rf_enable", False)
            and dino is not None
            and hasattr(dino, "merge_local_conv_rf_branches_")
        ):
            merged["dino_local_conv"] = dino.merge_local_conv_rf_branches_()
        dense_adapter = self._collect_dense_adapter()
        if (
            getattr(self.opt, "mfce_rf_enable", False)
            and dense_adapter is not None
            and hasattr(dense_adapter, "merge_rf_branches_")
        ):
            merged["mfce"] = dense_adapter.merge_rf_branches_()
        detector = self._collect_detector()
        if (
            getattr(self.opt, "decoder_rf_enable", False)
            and detector is not None
            and hasattr(detector, "merge_rf_branches_")
        ):
            merged["decoder"] = detector.merge_rf_branches_()
        pairlocal = self._collect_pairlocal()
        if (
            getattr(self.opt, "pairlocal_rf_enable", False)
            and pairlocal is not None
            and hasattr(pairlocal, "merge_rf_branches_")
        ):
            merged["pairlocal"] = pairlocal.merge_rf_branches_()
        return merged if merged else None

    def rf_diversity_loss(self):
        if (
            not getattr(self.opt, "mfce_rf_enable", False)
            or getattr(self.opt, "mfce_rf_diversity_weight", 0.0) <= 0
        ):
            return None
        dense_adapter = self._collect_dense_adapter()
        if dense_adapter is None or not hasattr(dense_adapter, "rf_diversity_loss"):
            return None
        return dense_adapter.rf_diversity_loss(
            margin=getattr(self.opt, "mfce_rf_diversity_margin", 1.0)
        )

    def update_lora_rank_search(self, epoch):
        return self.update_lora_rank_search_with_val(epoch, None)

    @torch.no_grad()
    def _evaluate_lora_counterfactual_metric(self, val_batches):
        if not val_batches:
            return None

        network = self._unwrap_model(self.model)
        was_training = network.training
        network.eval()

        confusion_matrix = None
        for batch in val_batches:
            x1 = batch["img1"].to(self.device, non_blocking=True)
            x2 = batch["img2"].to(self.device, non_blocking=True)
            label = batch["cd_label"].to(self.device, non_blocking=True)
            pred = network._forward(x1, x2)
            pred = torch.argmax(pred, dim=1)

            batch_confusion = get_confuse_matrix(
                num_classes=2,
                label_gts=label.detach().cpu().numpy(),
                label_preds=pred.detach().cpu().numpy(),
            )
            if confusion_matrix is None:
                confusion_matrix = batch_confusion
            else:
                confusion_matrix += batch_confusion

        if was_training:
            network.train()

        if confusion_matrix is None:
            return None

        scores = cm2score(confusion_matrix)
        metric_name = getattr(
            self.opt,
            "dino_lora_search_counterfactual_metric",
            "mean_iou1_f1",
        )
        if metric_name == "iou_1":
            score = float(scores.get("iou_1", 0.0))
        elif metric_name == "F1_1":
            score = float(scores.get("F1_1", 0.0))
        else:
            score = 0.5 * (
                float(scores.get("iou_1", 0.0)) + float(scores.get("F1_1", 0.0))
            )
        return {
            "score": score,
            "metric_name": metric_name,
            "metrics": scores,
        }

    def _set_batchnorm_eval(self, module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    def _evaluate_lora_probe_scores(self, probe_batches):
        if not probe_batches:
            return None

        network = self._unwrap_model(self.model)
        dino = getattr(getattr(network, "encoder", None), "dino", None)
        if dino is None or not hasattr(dino, "iter_searchable_lora_layers"):
            return None

        probe_layers = list(dino.iter_searchable_lora_layers())
        if not probe_layers:
            return None

        was_training = network.training
        network.train()
        network.apply(self._set_batchnorm_eval)
        network.zero_grad(set_to_none=True)

        score_sums = {layer.module_name: 0.0 for layer in probe_layers}
        score_counts = {layer.module_name: 0 for layer in probe_layers}

        with torch.enable_grad():
            for batch in probe_batches:
                network.zero_grad(set_to_none=True)

                x1 = batch["img1"].to(self.device, non_blocking=True)
                x2 = batch["img2"].to(self.device, non_blocking=True)
                label = batch["cd_label"].to(self.device, non_blocking=True).long()

                final_pred, preds = network(x1, x2)
                focal = self.focal(final_pred, label)
                dice = self.dice(final_pred, label)
                for pred in preds:
                    focal = focal + self.focal(pred, label)
                    dice = dice + 0.5 * self.dice(pred, label)
                loss = focal * self.opt.alpha + dice
                loss.backward()

                for layer in probe_layers:
                    if hasattr(layer, "probe_gradient_score"):
                        score = layer.probe_gradient_score()
                        if score is None:
                            continue
                        score_sums[layer.module_name] += score
                        score_counts[layer.module_name] += 1
                        continue

                    grad_a = layer.lora_A.weight.grad
                    grad_b = layer.lora_B.weight.grad
                    grad_sq = 0.0
                    param_count = 0
                    if grad_a is not None:
                        grad_sq += float(grad_a.detach().float().pow(2).sum().item())
                        param_count += grad_a.numel()
                    if grad_b is not None:
                        grad_sq += float(grad_b.detach().float().pow(2).sum().item())
                        param_count += grad_b.numel()
                    if param_count <= 0:
                        continue
                    score = (grad_sq ** 0.5) / (param_count ** 0.5)
                    score_sums[layer.module_name] += score
                    score_counts[layer.module_name] += 1

        network.zero_grad(set_to_none=True)
        if not was_training:
            network.eval()

        probe_scores = {}
        for module_name, score_sum in score_sums.items():
            count = max(1, score_counts[module_name])
            probe_scores[module_name] = score_sum / count
        if (
            getattr(self.opt, "dino_lora_search_spectral", False)
            and self.opt.is_main_process
            and probe_scores
        ):
            sorted_scores = sorted(
                probe_scores.items(),
                key=lambda item: item[1],
            )
            values = torch.tensor(
                [score for _, score in sorted_scores],
                dtype=torch.float32,
            )
            top_preview = ", ".join(
                f"{name}:{score:.3e}" for name, score in sorted_scores[-3:]
            )
            bottom_preview = ", ".join(
                f"{name}:{score:.3e}" for name, score in sorted_scores[:3]
            )
            print(
                "Spectral probe -> "
                f"mean={values.mean().item():.3e}, "
                f"std={values.std(unbiased=False).item():.3e}, "
                f"min={values.min().item():.3e}, "
                f"max={values.max().item():.3e}, "
                f"bottom=[{bottom_preview}], "
                f"top=[{top_preview}]"
            )
            self.last_spectral_probe_stats = {
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "bottom": sorted_scores[:3],
                "top": sorted_scores[-3:],
            }
        elif getattr(self.opt, "dino_lora_search_spectral", False):
            self.last_spectral_probe_stats = {}
        return probe_scores

    def sync_lora_rank_masks(self):
        if not (self.use_distributed and getattr(self.opt, "dino_lora_search", False)):
            return
        if not torch.distributed.is_initialized():
            return

        network = self._unwrap_model(self.model)
        dino = getattr(getattr(network, "encoder", None), "dino", None)
        if dino is None or not hasattr(dino, "iter_searchable_lora_layers"):
            return

        for layer in dino.iter_searchable_lora_layers():
            torch.distributed.broadcast(layer.rank_mask, src=0)
            layer.active_rank.fill_(int(layer.rank_mask.gt(0).sum().item()))
            if hasattr(layer, "counterfactual_confirm"):
                torch.distributed.broadcast(layer.counterfactual_confirm, src=0)

    def update_lora_rank_search_with_val(self, epoch, val_batches=None, probe_batches=None):
        self.current_epoch = int(epoch)
        if not getattr(self.opt, "dino_lora_search", False):
            return None

        network = self._unwrap_model(self.model)
        if not hasattr(network.encoder.dino, "update_lora_rank_search"):
            return None

        spectral_layers = self._collect_spectral_search_layers()
        pre_total_active_rank = int(
            sum(int(layer.active_rank.item()) for layer in spectral_layers)
        )
        summary = None
        eval_metric_fn = None
        probe_score_fn = None
        if (
            getattr(self.opt, "dino_lora_search_counterfactual", False)
            and val_batches
            and (not self.use_distributed or self.opt.is_main_process)
        ):
            eval_metric_fn = lambda: self._evaluate_lora_counterfactual_metric(val_batches)
        if (
            getattr(self.opt, "dino_lora_search_strategy", "classic") == "rfnext"
            and probe_batches
            and (not self.use_distributed or self.opt.is_main_process)
        ):
            probe_score_fn = lambda: self._evaluate_lora_probe_scores(probe_batches)

        if not self.use_distributed or self.opt.is_main_process:
            summary = network.encoder.dino.update_lora_rank_search(
                epoch,
                self.opt.num_epochs,
                eval_metric_fn=eval_metric_fn,
                probe_score_fn=probe_score_fn,
            )

        self.sync_lora_rank_masks()

        if summary and self.opt.is_main_process:
            summary["pre_total_active_rank"] = pre_total_active_rank
            preview = ", ".join(str(rank) for rank in summary["active_ranks"][:8])
            suffix = " ..." if len(summary["active_ranks"]) > 8 else ""
            group_preview = ", ".join(
                f"{group_name}:{rank}"
                for group_name, rank in sorted(summary.get("group_active_ranks", {}).items())
            )
            counterfactual_suffix = ""
            if summary.get("counterfactual", False):
                counterfactual_suffix = (
                    f", cf_tested={summary.get('counterfactual_tested', 0)}"
                    f", cf_safe={summary.get('counterfactual_accepted', 0)}"
                    f", cf_pruned={summary.get('counterfactual_pruned', 0)}"
                    f", cf_gap={summary.get('counterfactual_budget_gap', 0)}"
                    f", cf_metric={summary.get('counterfactual_metric_name', 'score')}"
                    f", cf_score={summary.get('counterfactual_baseline_score', float('nan')):.4f}"
                )
            probe_suffix = ""
            if summary.get("search_strategy") == "rfnext":
                probe_suffix = (
                    f", strategy=rfnext"
                    f", probe_blocks={summary.get('probe_selected_blocks', 0)}"
                    f", probe_layers={summary.get('probe_selected_layers', 0)}"
                    f", probe_modules={summary.get('probe_selected_modules', summary.get('probe_selected_layers', 0))}"
                    f", probe_epoch={summary.get('probe_epoch', -1)}"
                    f", probe_refresh={int(bool(summary.get('probe_refreshed', False)))}"
                )
            print(
                f"LoRA rank search -> budget={summary['budget_rank']}, "
                f"mode={summary.get('budget_mode', 'global')}, "
                f"total_active={summary['total_active_rank']}, "
                f"groups=[{group_preview}], "
                f"layer_ranks=[{preview}{suffix}]"
                f"{probe_suffix}"
                f"{counterfactual_suffix}"
            )
            if getattr(self.opt, "dino_lora_search_spectral", False):
                print(
                    "Spectral rank search -> "
                    f"pre_active={pre_total_active_rank}, "
                    f"post_active={summary.get('total_active_rank', pre_total_active_rank)}, "
                    f"delta_active={summary.get('total_active_rank', pre_total_active_rank) - pre_total_active_rank}"
                )
                self._log_spectral_search_state(f"epoch {epoch}")
                self.last_spectral_search_stats.update(
                    {
                        "epoch": int(epoch),
                        "pre_total_active_rank": int(pre_total_active_rank),
                        "post_total_active_rank": int(
                            summary.get("total_active_rank", pre_total_active_rank)
                        ),
                        "delta_total_active_rank": int(
                            summary.get("total_active_rank", pre_total_active_rank)
                            - pre_total_active_rank
                        ),
                        "search_strategy": summary.get("search_strategy", "classic"),
                        "probe_selected_blocks": int(
                            summary.get("probe_selected_blocks", 0)
                        ),
                        "probe_selected_modules": int(
                            summary.get("probe_selected_modules", 0)
                        ),
                    }
                )
            else:
                self.last_spectral_search_stats = {}
        return summary

    def forward(self, x1, x2, label):
        self.last_aux_losses = {}
        final_pred, preds = self.model(x1, x2)
        label = label.long()
        focal = self.focal(final_pred, label)
        dice = self.dice(final_pred, label)
        for i in range(len(preds)):
            focal += self.focal(preds[i], label)
            dice += 0.5 * self.dice(preds[i], label)

        rf_diversity = self.rf_diversity_loss()
        if rf_diversity is not None:
            self.last_aux_losses["rf_diversity"] = float(rf_diversity.detach().item())
            dice = dice + self.opt.mfce_rf_diversity_weight * rf_diversity

        if getattr(self.opt, "dino_lora_soft_gate", False):
            network = self._unwrap_model(self.model)
            dino = getattr(getattr(network, "encoder", None), "dino", None)
            budget_loss = None
            if dino is not None and hasattr(dino, "soft_gate_budget_loss"):
                budget_loss = dino.soft_gate_budget_loss(
                    target_ratio=self.opt.dino_lora_soft_gate_target_ratio,
                    mode=self.opt.dino_lora_soft_gate_budget_mode,
                )
            budget_lambda = self._soft_gate_budget_lambda(self.current_epoch)
            soft_gate_state = self._collect_soft_gate_debug_state() or {}
            self.last_aux_losses["soft_gate_budget_lambda"] = float(budget_lambda)
            self.last_aux_losses["soft_gate_effective_rank_total"] = float(
                soft_gate_state.get("effective_rank_total", 0.0)
            )
            self.last_aux_losses["soft_gate_expected_rank_ratio_mean"] = float(
                soft_gate_state.get("expected_rank_ratio_mean", 0.0)
            )
            if budget_loss is not None:
                self.last_aux_losses["soft_gate_budget_loss"] = float(
                    budget_loss.detach().item()
                )
                if budget_lambda > 0:
                    dice = dice + budget_lambda * budget_loss
            else:
                self.last_aux_losses["soft_gate_budget_loss"] = 0.0

        if getattr(self.opt, "cgla_temporal_reg_enable", False):
            reg_lambda = self._cgla_temporal_reg_lambda(self.current_epoch)
            reg_loss, reg_debug = self._cgla_temporal_regularization(label)
            self.last_aux_losses["cgla_temporal_reg_lambda"] = float(reg_lambda)
            self.last_aux_losses["cgla_temporal_reg_layers"] = float(
                reg_debug.get("cgla_temporal_reg_layers", 0.0)
            )
            self.last_aux_losses["cgla_temporal_reg_response_mean"] = float(
                reg_debug.get("cgla_temporal_reg_response_mean", 0.0)
            )
            self.last_aux_losses["cgla_temporal_reg_change_loss"] = float(
                reg_debug.get("cgla_temporal_reg_change_loss", 0.0)
            )
            self.last_aux_losses["cgla_temporal_reg_unchange_loss"] = float(
                reg_debug.get("cgla_temporal_reg_unchange_loss", 0.0)
            )
            self.last_aux_losses["cgla_temporal_reg_mask_mean"] = float(
                reg_debug.get("cgla_temporal_reg_mask_mean", 0.0)
            )
            self.last_aux_losses["cgla_temporal_reg_mask_nonzero_ratio"] = float(
                reg_debug.get("cgla_temporal_reg_mask_nonzero_ratio", 0.0)
            )
            if reg_loss is not None:
                self.last_aux_losses["cgla_temporal_reg_loss"] = float(
                    reg_loss.detach().item()
                )
                if reg_lambda > 0:
                    dice = dice + reg_lambda * reg_loss
            else:
                self.last_aux_losses["cgla_temporal_reg_loss"] = 0.0

        detector_debug = self._collect_detector_debug_state() or {}
        self.last_aux_losses["decoder_bifpn_enable"] = float(
            detector_debug.get("decoder_bifpn_enable", 0.0)
        )
        self.last_aux_losses["decoder_cgla_bifpn_enable"] = float(
            detector_debug.get("decoder_cgla_bifpn_enable", 0.0)
        )
        self.last_aux_losses["decoder_bifpn_weight_mean"] = float(
            detector_debug.get("decoder_bifpn_weight_mean", 0.0)
        )
        self.last_aux_losses["decoder_cgla_bifpn_prior_scale_mean"] = float(
            detector_debug.get("decoder_cgla_bifpn_prior_scale_mean", 0.0)
        )

        return final_pred, focal, dice

    @torch.inference_mode()
    def inference(self, x1, x2):
        pred = self._unwrap_model(self.model)._forward(x1, x2)
        return pred

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = "%s_%s_best.pth" % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            existing = []
            if os.path.isdir(self.save_dir):
                existing = sorted(
                    file_name
                    for file_name in os.listdir(self.save_dir)
                    if file_name.endswith(".pth")
                )
            raise FileNotFoundError(
                "Expected checkpoint was not found.\n"
                f"Expected: {save_path}\n"
                f"Available in save_dir: {existing if existing else 'None'}"
            )
        else:
            checkpoint = torch.load(
                save_path, map_location="cpu", weights_only=True
            )
            self._unwrap_model(network).load_state_dict(checkpoint["network"], strict=False)
            print("load pre-trained")

    def _serializable_opt_dict(self):
        serializable = {}
        simple_types = (str, int, float, bool, type(None))
        for key, value in vars(self.opt).items():
            if isinstance(value, simple_types):
                serializable[key] = value
            elif isinstance(value, (list, tuple)):
                if all(isinstance(item, simple_types) for item in value):
                    serializable[key] = list(value)
            elif isinstance(value, dict):
                if all(
                    isinstance(k, str) and isinstance(v, simple_types)
                    for k, v in value.items()
                ):
                    serializable[key] = dict(value)
        return serializable

    def save_checkpoint(self, path, epoch=None, scores=None, extra=None):
        state_dict = {
            key: value.detach().cpu()
            for key, value in self._unwrap_model(self.model).state_dict().items()
        }
        checkpoint = {
            "network": state_dict,
            "model_state_dict": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scores": scores,
            "epoch": epoch,
            "extra": extra,
            "opt": self._serializable_opt_dict(),
        }
        if self.schedular is not None:
            checkpoint["scheduler_state_dict"] = self.schedular.state_dict()
        torch.save(checkpoint, path)

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = "%s_%s_best.pth" % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        self.save_checkpoint(save_path)

    def save(self, model_name, backbone):
        self.save_ckpt(self.model, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    if opt.is_main_process:
        print("model [%s] was created" % model.name())

    return model.cuda()
