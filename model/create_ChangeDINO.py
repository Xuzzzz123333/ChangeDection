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

        self.model = get_model(
            backbone_name=opt.backbone,
            fpn_name=opt.fpn,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            n_layers=opt.n_layers,
            extract_ids=opt.extract_ids,
            dino_lora=opt.dino_lora,
            dino_dora=opt.dino_dora,
            dino_lora_r=opt.dino_lora_r,
            dino_lora_alpha=opt.dino_lora_alpha,
            dino_lora_dropout=opt.dino_lora_dropout,
            dino_lora_search=opt.dino_lora_search,
            dino_lora_r_target=opt.dino_lora_r_target,
            dino_lora_alpha_over_r=opt.dino_lora_alpha_over_r,
            dino_lora_search_warmup_epochs=opt.dino_lora_search_warmup_epochs,
            dino_lora_search_interval=opt.dino_lora_search_interval,
            dino_lora_search_ema_decay=opt.dino_lora_search_ema_decay,
            dino_lora_search_score_norm=opt.dino_lora_search_score_norm,
            dino_lora_search_grad_weight=opt.dino_lora_search_grad_weight,
            dino_lora_search_budget_mode=opt.dino_lora_search_budget_mode,
            dino_lora_search_group_weights=opt.dino_lora_search_group_weights,
            dino_lora_search_counterfactual=opt.dino_lora_search_counterfactual,
            dino_lora_search_counterfactual_val_batches=opt.dino_lora_search_counterfactual_val_batches,
            dino_lora_search_counterfactual_max_candidates=opt.dino_lora_search_counterfactual_max_candidates,
            dino_lora_search_counterfactual_delta=opt.dino_lora_search_counterfactual_delta,
            dino_lora_search_counterfactual_patience=opt.dino_lora_search_counterfactual_patience,
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
            acpc_enable=opt.acpc_enable,
            acpc_stage_modes=opt.acpc_stage_modes,
            acpc_hidden_ratio=opt.acpc_hidden_ratio,
            acpc_norm_groups=opt.acpc_norm_groups,
            acpc_residual_scale=opt.acpc_residual_scale,
        )
        self._log_trainable_parameters()
        if opt.load_pretrain:
            self.load_ckpt(self.model, None, opt.name, opt.backbone)
        should_merge_rf = opt.load_pretrain and (
            (
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
                if ".lora_" in name or ".dora_" in name
            ]
            print(f"trainable DINO tensors: {len(dino_trainable)}")
            print(f"trainable DINO parameters total: {sum(numel for _, numel in dino_trainable)}")
            adapter_label = "DoRA" if getattr(self.opt, "dino_dora", False) else "LoRA"
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

    def _log_rf_states(self, prefix="current"):
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

    def configure_rf_search(self, steps_per_epoch: int):
        dense_adapter = self._collect_dense_adapter()
        detector = self._collect_detector()
        summaries = {}

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
        return summaries if summaries else None

    def merge_rf_branches(self):
        merged = {}
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

    def update_lora_rank_search_with_val(self, epoch, val_batches=None):
        if not getattr(self.opt, "dino_lora_search", False):
            return None

        network = self._unwrap_model(self.model)
        if not hasattr(network.encoder.dino, "update_lora_rank_search"):
            return None

        summary = None
        eval_metric_fn = None
        if (
            getattr(self.opt, "dino_lora_search_counterfactual", False)
            and val_batches
            and (not self.use_distributed or self.opt.is_main_process)
        ):
            eval_metric_fn = lambda: self._evaluate_lora_counterfactual_metric(val_batches)

        if not self.use_distributed or self.opt.is_main_process:
            summary = network.encoder.dino.update_lora_rank_search(
                epoch,
                self.opt.num_epochs,
                eval_metric_fn=eval_metric_fn,
            )

        self.sync_lora_rank_masks()

        if summary and self.opt.is_main_process:
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
            print(
                f"LoRA rank search -> budget={summary['budget_rank']}, "
                f"mode={summary.get('budget_mode', 'global')}, "
                f"total_active={summary['total_active_rank']}, "
                f"groups=[{group_preview}], "
                f"layer_ranks=[{preview}{suffix}]"
                f"{counterfactual_suffix}"
            )
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
                save_path, map_location=self.device, weights_only=True
            )
            self._unwrap_model(network).load_state_dict(checkpoint["network"], strict=False)
            print("load pre-trained")

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = "%s_%s_best.pth" % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        state_dict = {
            key: value.detach().cpu()
            for key, value in self._unwrap_model(network).state_dict().items()
        }
        torch.save(
            {
                "network": state_dict,
                "optimizer": optimizer.state_dict(),
            },
            save_path,
        )

    def save(self, model_name, backbone):
        self.save_ckpt(self.model, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    if opt.is_main_process:
        print("model [%s] was created" % model.name())

    return model.cuda()
