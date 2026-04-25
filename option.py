import argparse
import os
import torch


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument(
            "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0. use -1 for CPU"
        )
        self.parser.add_argument("--name", type=str, default="WHU")
        self.parser.add_argument(
            "--dataroot", type=str, default="/ssddd/chingheng/CD-Dataset"
        )
        self.parser.add_argument("--dataset", type=str, default="WHU-CD")
        self.parser.add_argument(
            "--data_mode",
            type=str,
            default="original",
            choices=["original", "custom_patch"],
            help="dataset pipeline to use",
        )
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )
        
        self.parser.add_argument(
            "--save_test", action="store_true"
        )
        self.parser.add_argument(
            "--result_dir", type=str, default="./results", help="results are saved here"
        )
        self.parser.add_argument(
            "--vis_path", type=str, default="vis", help="results are saved here"
        )
        self.parser.add_argument(
            "--vis_interval",
            type=int,
            default=1,
            help="save train/val visualizations every N epochs; set to 0 to disable",
        )
        self.parser.add_argument("--load_pretrain", action='store_true')
        self.parser.add_argument("--use_morph", action='store_true')

        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument("--backbone", type=str, default="mobilenetv2")
        self.parser.add_argument("--fpn", type=str, default="fpn")
        self.parser.add_argument("--fpn_channels", type=int, default=128)
        self.parser.add_argument("--deform_groups", type=int, default=4)
        self.parser.add_argument("--gamma_mode", type=str, default="SE")
        self.parser.add_argument("--beta_mode", type=str, default="contextgatedconv")
        self.parser.add_argument('--n_layers', nargs='+', type=int, default=[1, 1, 1, 1])
        self.parser.add_argument('--extract_ids', nargs='+', type=int, default=[5, 11, 17, 23])
        self.parser.add_argument("--alpha", type=float, default=0.25)
        self.parser.add_argument("--gamma", type=int, default=4, help="gamma for Focal loss")

        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument(
            "--grad_accum_steps",
            type=int,
            default=1,
            help="number of gradient accumulation steps; effective batch = batch_size * grad_accum_steps",
        )
        self.parser.add_argument(
            "--image_size",
            type=int,
            default=256,
            help="input size used by the custom_patch data mode",
        )
        self.parser.add_argument("--num_epochs", type=int, default=100)
        self.parser.add_argument("--num_workers", type=int, default=4, help="#threads for loading data")
        self.parser.add_argument("--lr", type=float, default=5e-4)
        self.parser.add_argument("--weight_decay", type=float, default=5e-4)
        self.parser.add_argument("--dino_lora", action="store_true")
        self.parser.add_argument(
            "--dino_dora",
            action="store_true",
            help="use fixed-rank DoRA adapters on DINOv3 linear layers instead of LoRA",
        )
        self.parser.add_argument(
            "--dino_local_conv_enable",
            action="store_true",
            help="insert a lightweight local convolution branch into selected DINO transformer blocks",
        )
        self.parser.add_argument(
            "--dino_local_conv_blocks",
            nargs="+",
            type=int,
            default=[5, 11, 17, 23],
            help="DINO block indices that receive the local convolution branch",
        )
        self.parser.add_argument(
            "--dino_local_conv_kernel_size",
            type=int,
            default=3,
            help="odd kernel size used by the DINO local convolution branch",
        )
        self.parser.add_argument(
            "--dino_local_conv_init_scale",
            type=float,
            default=0.0,
            help="initial residual scale for the DINO local convolution branch; 0 keeps the pretrained backbone unchanged at startup",
        )
        self.parser.add_argument(
            "--dino_local_conv_change_aware_enable",
            action="store_true",
            help="upgrade the DINO internal local branch into a bi-temporal change-aware paired adapter with shared gating and signed change residuals",
        )
        self.parser.add_argument(
            "--dino_local_conv_change_hidden_ratio",
            type=float,
            default=0.5,
            help="hidden expansion ratio used by the change-aware relation encoder inside the DINO local branch",
        )
        self.parser.add_argument(
            "--dino_local_conv_change_norm_groups",
            type=int,
            default=8,
            help="group count used by change-aware normalization inside the DINO local branch",
        )
        self.parser.add_argument(
            "--dino_local_conv_change_residual_scale",
            type=float,
            default=0.05,
            help="initial global residual scale for the change-aware DINO local branch",
        )
        self.parser.add_argument(
            "--dino_local_conv_change_delta_scale",
            type=float,
            default=0.05,
            help="initial signed delta residual scale for the change-aware DINO local branch",
        )
        self.parser.add_argument(
            "--dino_local_conv_change_mixer_enable",
            action="store_true",
            help="add a lightweight directional long-range mixer inside the change-aware DINO relation branch",
        )
        self.parser.add_argument(
            "--dino_local_conv_change_mixer_kernel_size",
            type=int,
            default=7,
            help="odd kernel size used by the lightweight directional relation mixer",
        )
        self.parser.add_argument(
            "--dino_local_conv_change_mixer_residual_scale",
            type=float,
            default=1.0,
            help="initial residual scale used by the lightweight directional relation mixer",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_enable",
            action="store_true",
            help="replace the DINO local-conv branch depthwise convolution with RF-Next receptive-field search",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_mode",
            type=str,
            default="rfsearch",
            choices=["rfsearch", "rfsingle", "rfmultiple", "rfmerge"],
            help="RF-Next mode used inside the DINO local-conv branch depthwise convolution",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_num_branches",
            type=int,
            default=3,
            help="number of local dilation candidates maintained by each DINO local-conv RF branch",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_expand_rate",
            type=float,
            default=0.5,
            help="local expansion ratio used by DINO local-conv RF search",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_min_dilation",
            type=int,
            default=1,
            help="minimum dilation allowed by DINO local-conv RF search",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_max_dilations",
            nargs="+",
            type=int,
            default=None,
            help="optional per-block max dilations for DINO local-conv RF search; provide one value or one per selected DINO block",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_search_interval",
            type=int,
            default=100,
            help="forward-step interval between DINO local-conv RF estimate-expand updates",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_max_search_step",
            type=int,
            default=8,
            help="maximum number of DINO local-conv RF local search refinements; 0 disables iterative updates",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_init_weight",
            type=float,
            default=0.01,
            help="initial branch weight used by DINO local-conv RF search",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_schedule_mode",
            type=str,
            default="epoch",
            choices=["manual", "epoch"],
            help="manual step schedule or epoch-aware DINO local-conv RF search schedule",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_search_warmup_epochs",
            type=int,
            default=0,
            help="delay DINO local-conv RF search updates for the first N epochs",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_search_epochs",
            type=int,
            default=20,
            help="number of epochs allocated to DINO local-conv RF refinement in epoch schedule mode; <=0 uses the remaining training epochs",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_log_interval",
            type=int,
            default=5,
            help="log DINO local-conv RF states every N epochs during training; set to 0 to disable",
        )
        self.parser.add_argument(
            "--dino_local_conv_rf_merge_on_eval",
            action="store_true",
            help="merge searched DINO local-conv RF branches into a single equivalent convolution after loading checkpoints for evaluation",
        )
        self.parser.add_argument("--dino_lora_r", type=int, default=8)
        self.parser.add_argument("--dino_lora_alpha", type=int, default=16)
        self.parser.add_argument("--dino_lora_dropout", type=float, default=0.05)
        self.parser.add_argument(
            "--dino_lora_search",
            action="store_true",
            help="enable importance-based rank search on top of DINO LoRA",
        )
        self.parser.add_argument(
            "--dino_lora_r_target",
            type=int,
            default=4,
            help="target average LoRA rank after importance-based pruning; can be 0",
        )
        self.parser.add_argument(
            "--dino_lora_alpha_over_r",
            type=float,
            default=1.0,
            help="fixed alpha/r scaling used by searchable LoRA",
        )
        self.parser.add_argument(
            "--dino_lora_search_warmup_epochs",
            type=int,
            default=5,
            help="keep the max LoRA rank for the first N epochs before pruning",
        )
        self.parser.add_argument(
            "--dino_lora_search_interval",
            type=int,
            default=1,
            help="update searchable LoRA ranks every N epochs",
        )
        self.parser.add_argument(
            "--dino_lora_search_ema_decay",
            type=float,
            default=0.9,
            help="EMA decay used to smooth searchable LoRA importance scores",
        )
        self.parser.add_argument(
            "--dino_lora_search_score_norm",
            type=str,
            default="median",
            choices=["none", "median", "zscore"],
            help="group-wise normalization applied before searchable LoRA ranking",
        )
        self.parser.add_argument(
            "--dino_lora_search_grad_weight",
            type=float,
            default=0.5,
            help="strength of gradient-sensitivity modulation for searchable LoRA scores",
        )
        self.parser.add_argument(
            "--dino_lora_search_spectral",
            action="store_true",
            help="replace searchable LoRA channels with frozen SVD directions and trainable singular-value scales",
        )
        self.parser.add_argument(
            "--dino_lora_spectral_prior_power",
            type=float,
            default=0.5,
            help="strength of pretrained singular-value prior used by spectral searchable LoRA",
        )
        self.parser.add_argument(
            "--dino_lora_spectral_uncertainty_weight",
            type=float,
            default=0.5,
            help="AdaLoRA-style uncertainty weight applied to spectral searchable LoRA scores",
        )
        self.parser.add_argument(
            "--dino_lora_spectral_init_scale",
            type=float,
            default=0.0,
            help="initial singular scale for spectral searchable LoRA; 0 keeps the frozen DINO layer unchanged at startup",
        )
        self.parser.add_argument(
            "--dino_lora_search_budget_mode",
            type=str,
            default="grouped",
            choices=["global", "grouped"],
            help="allocate searchable LoRA ranks globally or separately per position group",
        )
        self.parser.add_argument(
            "--dino_lora_search_group_weights",
            nargs="+",
            default=[
                "attn.qkv=1.0",
                "attn.proj=1.0",
                "mlp.fc1=1.0",
                "mlp.fc2=1.0",
            ],
            help="per-group budget weights for searchable LoRA, formatted as group=value; supports base groups and optional depth-bucket groups",
        )
        self.parser.add_argument(
            "--dino_lora_search_depth_buckets",
            type=int,
            default=3,
            help="number of depth buckets used for grouped LoRA rank search; set to 1 to recover position-only grouping",
        )
        self.parser.add_argument(
            "--dino_lora_search_strategy",
            type=str,
            default="classic",
            choices=["classic", "rfnext"],
            help="classic grouped rank search or an RF-Next-style coarse-to-fine rank search",
        )
        self.parser.add_argument(
            "--dino_lora_search_probe_batches",
            type=int,
            default=5,
            help="number of representative training batches used by the RF-style LoRA layer probe",
        )
        self.parser.add_argument(
            "--dino_lora_search_probe_refresh_interval",
            type=int,
            default=10,
            help="refresh the RF-style LoRA probe every N search epochs after initialization; 0 keeps one-shot probing",
        )
        self.parser.add_argument(
            "--dino_lora_search_probe_score_norm",
            type=str,
            default="zscore",
            choices=["none", "median", "zscore"],
            help="normalization applied to probe scores before quantile-based RF-style LoRA prior construction",
        )
        self.parser.add_argument(
            "--dino_lora_search_probe_keep_ratio",
            type=float,
            default=0.5,
            help="fraction of transformer blocks kept after the probe stage in RF-style LoRA search",
        )
        self.parser.add_argument(
            "--dino_lora_search_probe_module_keep_ratio",
            type=float,
            default=0.75,
            help="fraction of LoRA target modules kept inside each selected transformer block during RF-style probe selection",
        )
        self.parser.add_argument(
            "--dino_lora_search_rf_delta",
            type=int,
            default=2,
            help="local rank search radius around the current center for RF-style LoRA search",
        )
        self.parser.add_argument(
            "--dino_lora_search_rf_temperature",
            type=float,
            default=1.0,
            help="softmax temperature used when converting local rank candidate utilities into expected ranks",
        )
        self.parser.add_argument(
            "--dino_lora_search_counterfactual",
            action="store_true",
            help="use a small validation set to counterfactually test candidate LoRA rank deletions before pruning",
        )
        self.parser.add_argument(
            "--dino_lora_search_counterfactual_val_batches",
            type=int,
            default=4,
            help="number of representative validation batches used by counterfactual LoRA rank testing",
        )
        self.parser.add_argument(
            "--dino_lora_search_counterfactual_metric",
            type=str,
            default="mean_iou1_f1",
            choices=["iou_1", "F1_1", "mean_iou1_f1"],
            help="validation metric used to decide whether a candidate LoRA rank direction is safe to prune",
        )
        self.parser.add_argument(
            "--dino_lora_search_counterfactual_max_candidates",
            type=int,
            default=64,
            help="maximum number of low-score LoRA rank directions tested per search update",
        )
        self.parser.add_argument(
            "--dino_lora_search_counterfactual_delta",
            type=float,
            default=0.0,
            help="only prune a tested LoRA direction when dropping it reduces the chosen validation metric by at most this threshold",
        )
        self.parser.add_argument(
            "--dino_lora_search_counterfactual_patience",
            type=int,
            default=1,
            help="number of consecutive counterfactual confirmations required before pruning a LoRA direction",
        )
        self.parser.add_argument(
            "--mfce_enable",
            action="store_true",
            help="replace the per-layer dense adapter with MFCE-style multi-layer fusion and ASPP context enhancement",
        )
        self.parser.add_argument(
            "--mfce_mid_dim",
            type=int,
            default=256,
            help="hidden width used by the MFCE dense branch",
        )
        self.parser.add_argument(
            "--mfce_aspp_rates",
            nargs="+",
            type=int,
            default=[1, 2, 4, 8],
            help="dilation rates used by the MFCE ASPP context module",
        )
        self.parser.add_argument(
            "--mfce_rf_enable",
            action="store_true",
            help="replace fixed ASPP depthwise dilations with RF-Next style local receptive-field search",
        )
        self.parser.add_argument(
            "--mfce_rf_mode",
            type=str,
            default="rfsearch",
            choices=["rfsearch", "rfsingle", "rfmultiple", "rfmerge"],
            help="RF-Next mode used inside each MFCE ASPP branch",
        )
        self.parser.add_argument(
            "--mfce_rf_num_branches",
            type=int,
            default=3,
            help="number of local dilation candidates maintained by each RF-Next ASPP branch",
        )
        self.parser.add_argument(
            "--mfce_rf_expand_rate",
            type=float,
            default=0.5,
            help="local expansion ratio used by RF-Next receptive-field search",
        )
        self.parser.add_argument(
            "--mfce_rf_min_dilation",
            type=int,
            default=1,
            help="minimum dilation allowed by RF-Next ASPP search",
        )
        self.parser.add_argument(
            "--mfce_rf_max_dilations",
            nargs="+",
            type=int,
            default=None,
            help="optional per-branch max dilations for MFCE RF search; provide one value or one per ASPP branch",
        )
        self.parser.add_argument(
            "--mfce_rf_search_interval",
            type=int,
            default=100,
            help="forward-step interval between RF-Next estimate-expand updates",
        )
        self.parser.add_argument(
            "--mfce_rf_max_search_step",
            type=int,
            default=8,
            help="maximum number of RF-Next local search refinements; 0 disables iterative updates",
        )
        self.parser.add_argument(
            "--mfce_rf_init_weight",
            type=float,
            default=0.01,
            help="initial branch weight used by RF-Next ASPP search",
        )
        self.parser.add_argument(
            "--mfce_rf_schedule_mode",
            type=str,
            default="epoch",
            choices=["manual", "epoch"],
            help="manual step schedule or epoch-aware RF search schedule",
        )
        self.parser.add_argument(
            "--mfce_rf_search_warmup_epochs",
            type=int,
            default=0,
            help="delay RF search updates for the first N epochs",
        )
        self.parser.add_argument(
            "--mfce_rf_search_epochs",
            type=int,
            default=20,
            help="number of epochs allocated to RF search refinement in epoch schedule mode; <=0 uses the remaining training epochs",
        )
        self.parser.add_argument(
            "--mfce_rf_diversity_weight",
            type=float,
            default=0.05,
            help="weight of the ASPP branch diversity regularizer for RF search",
        )
        self.parser.add_argument(
            "--mfce_rf_diversity_margin",
            type=float,
            default=1.0,
            help="minimum expected dilation gap enforced between RF-ASPP branches",
        )
        self.parser.add_argument(
            "--mfce_rf_log_interval",
            type=int,
            default=5,
            help="log RF branch states every N epochs during training; set to 0 to disable",
        )
        self.parser.add_argument(
            "--mfce_rf_merge_on_eval",
            action="store_true",
            help="merge searched RF branches into a single equivalent convolution after loading checkpoints for evaluation",
        )
        self.parser.add_argument(
            "--decoder_rf_enable",
            action="store_true",
            help="replace FuseGated mix convolutions with RF-Next style receptive-field search",
        )
        self.parser.add_argument(
            "--decoder_rf_mode",
            type=str,
            default="rfsearch",
            choices=["rfsearch", "rfsingle", "rfmultiple", "rfmerge"],
            help="RF-Next mode used inside each decoder FuseGated mix convolution",
        )
        self.parser.add_argument(
            "--decoder_rf_num_branches",
            type=int,
            default=3,
            help="number of local dilation candidates maintained by each decoder RF branch",
        )
        self.parser.add_argument(
            "--decoder_rf_expand_rate",
            type=float,
            default=0.5,
            help="local expansion ratio used by decoder RF search",
        )
        self.parser.add_argument(
            "--decoder_rf_min_dilation",
            type=int,
            default=1,
            help="minimum dilation allowed by decoder RF search",
        )
        self.parser.add_argument(
            "--decoder_rf_max_dilations",
            nargs="+",
            type=int,
            default=None,
            help="optional per-stage max dilations for decoder RF search; provide one value or one per FuseGated stage",
        )
        self.parser.add_argument(
            "--decoder_rf_search_interval",
            type=int,
            default=100,
            help="forward-step interval between decoder RF estimate-expand updates",
        )
        self.parser.add_argument(
            "--decoder_rf_max_search_step",
            type=int,
            default=8,
            help="maximum number of decoder RF local search refinements; 0 disables iterative updates",
        )
        self.parser.add_argument(
            "--decoder_rf_init_weight",
            type=float,
            default=0.01,
            help="initial branch weight used by decoder RF search",
        )
        self.parser.add_argument(
            "--decoder_rf_schedule_mode",
            type=str,
            default="epoch",
            choices=["manual", "epoch"],
            help="manual step schedule or epoch-aware decoder RF search schedule",
        )
        self.parser.add_argument(
            "--decoder_rf_search_warmup_epochs",
            type=int,
            default=0,
            help="delay decoder RF search updates for the first N epochs",
        )
        self.parser.add_argument(
            "--decoder_rf_search_epochs",
            type=int,
            default=20,
            help="number of epochs allocated to decoder RF refinement in epoch schedule mode; <=0 uses the remaining training epochs",
        )
        self.parser.add_argument(
            "--decoder_rf_log_interval",
            type=int,
            default=5,
            help="log decoder RF states every N epochs during training; set to 0 to disable",
        )
        self.parser.add_argument(
            "--decoder_rf_merge_on_eval",
            action="store_true",
            help="merge searched decoder RF branches into a single equivalent convolution after loading checkpoints for evaluation",
        )
        self.parser.add_argument(
            "--decoder_pred_guided_enable",
            action="store_true",
            help="feed coarse decoder predictions back into the next top-down fusion gate for change-aware refinement",
        )
        self.parser.add_argument(
            "--decoder_pred_guided_mode",
            type=str,
            default="prob_uncertainty_boundary",
            choices=["prob", "prob_uncertainty", "prob_uncertainty_boundary"],
            help="decoder guidance cues injected into top-down fusion: coarse probability only, probability+uncertainty, or probability+uncertainty+boundary",
        )
        self.parser.add_argument(
            "--dino_temporal_exchange_enable",
            action="store_true",
            help="enable cross-temporal exchange on raw DINO features before dense adaptation",
        )
        self.parser.add_argument(
            "--dino_temporal_exchange_mode",
            type=str,
            default="layer",
            choices=[
                "none",
                "layer",
                "rand_layer",
                "channel",
                "rand_channel",
                "spatial",
                "rand_spatial",
            ],
            help="temporal exchange mode applied to paired DINO features",
        )
        self.parser.add_argument(
            "--dino_temporal_exchange_thresh",
            type=float,
            default=0.5,
            help="probability threshold used by random temporal exchange modes",
        )
        self.parser.add_argument(
            "--dino_temporal_exchange_p",
            type=int,
            default=2,
            help="stride used by deterministic channel or spatial temporal exchange",
        )
        self.parser.add_argument(
            "--dino_temporal_exchange_layers",
            nargs="+",
            type=int,
            default=[0, 1, 2, 3],
            help="DINO feature indices that participate in temporal exchange",
        )
        self.parser.add_argument(
            "--pairlocal_enable",
            action="store_true",
            help="enable pair-aware interaction on the local FPN branch before DINO fusion",
        )
        self.parser.add_argument(
            "--pairlocal_stage_modes",
            nargs="+",
            default=["full", "full", "lite", "lite"],
            choices=["none", "lite", "full"],
            help="per-level pair-local modes for p2 p3 p4 p5",
        )
        self.parser.add_argument(
            "--pairlocal_hidden_ratio",
            type=float,
            default=1.0,
            help="hidden width ratio used inside pair-local interaction blocks",
        )
        self.parser.add_argument(
            "--pairlocal_norm_groups",
            type=int,
            default=8,
            help="group count used by pair-local GroupNorm layers",
        )
        self.parser.add_argument(
            "--pairlocal_residual_scale",
            type=float,
            default=0.1,
            help="initial residual scale for pair-local feature updates",
        )
        self.parser.add_argument(
            "--pairlocal_rf_enable",
            action="store_true",
            help="replace full-mode PairLocal context depthwise convolutions with RF-Next style receptive-field search",
        )
        self.parser.add_argument(
            "--pairlocal_rf_mode",
            type=str,
            default="rfsearch",
            choices=["rfsearch", "rfsingle", "rfmultiple", "rfmerge"],
            help="RF-Next mode used inside full-mode PairLocal context convolutions",
        )
        self.parser.add_argument(
            "--pairlocal_rf_num_branches",
            type=int,
            default=3,
            help="number of local dilation candidates maintained by each PairLocal RF branch",
        )
        self.parser.add_argument(
            "--pairlocal_rf_expand_rate",
            type=float,
            default=0.5,
            help="local expansion ratio used by PairLocal RF search",
        )
        self.parser.add_argument(
            "--pairlocal_rf_min_dilation",
            type=int,
            default=1,
            help="minimum dilation allowed by PairLocal RF search",
        )
        self.parser.add_argument(
            "--pairlocal_rf_max_dilations",
            nargs="+",
            type=int,
            default=None,
            help="optional per-stage max dilations for PairLocal RF search; provide one value or one per p2-p5 stage",
        )
        self.parser.add_argument(
            "--pairlocal_rf_search_interval",
            type=int,
            default=100,
            help="forward-step interval between PairLocal RF estimate-expand updates",
        )
        self.parser.add_argument(
            "--pairlocal_rf_max_search_step",
            type=int,
            default=8,
            help="maximum number of PairLocal RF local search refinements; 0 disables iterative updates",
        )
        self.parser.add_argument(
            "--pairlocal_rf_init_weight",
            type=float,
            default=0.01,
            help="initial branch weight used by PairLocal RF search",
        )
        self.parser.add_argument(
            "--pairlocal_rf_schedule_mode",
            type=str,
            default="epoch",
            choices=["manual", "epoch"],
            help="manual step schedule or epoch-aware PairLocal RF search schedule",
        )
        self.parser.add_argument(
            "--pairlocal_rf_search_warmup_epochs",
            type=int,
            default=0,
            help="delay PairLocal RF search updates for the first N epochs",
        )
        self.parser.add_argument(
            "--pairlocal_rf_search_epochs",
            type=int,
            default=20,
            help="number of epochs allocated to PairLocal RF refinement in epoch schedule mode; <=0 uses the remaining training epochs",
        )
        self.parser.add_argument(
            "--pairlocal_rf_log_interval",
            type=int,
            default=5,
            help="log PairLocal RF states every N epochs during training; set to 0 to disable",
        )
        self.parser.add_argument(
            "--pairlocal_rf_merge_on_eval",
            action="store_true",
            help="merge searched PairLocal RF branches into a single equivalent convolution after loading checkpoints for evaluation",
        )
        self.parser.add_argument(
            "--acpc_enable",
            action="store_true",
            help="enable adaptive change prior construction before the decoder",
        )
        self.parser.add_argument(
            "--acpc_stage_modes",
            nargs="+",
            default=["full", "lite", "none", "none"],
            choices=["none", "lite", "full"],
            help="per-level ACPC modes for p2 p3 p4 p5",
        )
        self.parser.add_argument(
            "--acpc_hidden_ratio",
            type=float,
            default=0.5,
            help="hidden width ratio used inside ACPC blocks",
        )
        self.parser.add_argument(
            "--acpc_norm_groups",
            type=int,
            default=8,
            help="group count used by ACPC GroupNorm layers",
        )
        self.parser.add_argument(
            "--acpc_residual_scale",
            type=float,
            default=0.05,
            help="initial modulation strength for ACPC reweighting",
        )

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        self.opt.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.opt.rank = int(os.environ.get("RANK", 0))
        self.opt.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.opt.distributed = self.opt.world_size > 1
        self.opt.is_main_process = self.opt.rank == 0

        if len(self.opt.pairlocal_stage_modes) != 4:
            raise ValueError(
                "--pairlocal_stage_modes expects exactly 4 values for p2 p3 p4 p5."
            )
        if len(self.opt.acpc_stage_modes) != 4:
            raise ValueError(
                "--acpc_stage_modes expects exactly 4 values for p2 p3 p4 p5."
            )
        if self.opt.vis_interval < 0:
            raise ValueError("--vis_interval must be >= 0.")
        if self.opt.dino_lora and self.opt.dino_dora:
            raise ValueError("--dino_lora and --dino_dora are mutually exclusive.")
        if not self.opt.dino_local_conv_blocks:
            raise ValueError("--dino_local_conv_blocks expects at least one block index.")
        if any(block_index < 0 for block_index in self.opt.dino_local_conv_blocks):
            raise ValueError("--dino_local_conv_blocks must use non-negative integers.")
        if self.opt.dino_local_conv_kernel_size <= 0 or self.opt.dino_local_conv_kernel_size % 2 == 0:
            raise ValueError("--dino_local_conv_kernel_size must be a positive odd integer.")
        if self.opt.dino_local_conv_change_aware_enable and not self.opt.dino_local_conv_enable:
            raise ValueError(
                "--dino_local_conv_change_aware_enable requires --dino_local_conv_enable to be enabled."
            )
        if self.opt.dino_local_conv_change_hidden_ratio <= 0:
            raise ValueError("--dino_local_conv_change_hidden_ratio must be > 0.")
        if self.opt.dino_local_conv_change_norm_groups <= 0:
            raise ValueError("--dino_local_conv_change_norm_groups must be > 0.")
        if self.opt.dino_local_conv_change_residual_scale < 0:
            raise ValueError("--dino_local_conv_change_residual_scale must be >= 0.")
        if self.opt.dino_local_conv_change_delta_scale < 0:
            raise ValueError("--dino_local_conv_change_delta_scale must be >= 0.")
        if (
            self.opt.dino_local_conv_change_mixer_enable
            and not self.opt.dino_local_conv_change_aware_enable
        ):
            raise ValueError(
                "--dino_local_conv_change_mixer_enable requires --dino_local_conv_change_aware_enable to be enabled."
            )
        if (
            self.opt.dino_local_conv_change_mixer_kernel_size <= 0
            or self.opt.dino_local_conv_change_mixer_kernel_size % 2 == 0
        ):
            raise ValueError(
                "--dino_local_conv_change_mixer_kernel_size must be a positive odd integer."
            )
        if self.opt.dino_local_conv_change_mixer_residual_scale < 0:
            raise ValueError(
                "--dino_local_conv_change_mixer_residual_scale must be >= 0."
            )
        if self.opt.dino_local_conv_rf_enable and not self.opt.dino_local_conv_enable:
            raise ValueError("--dino_local_conv_rf_enable requires --dino_local_conv_enable to be enabled.")
        if self.opt.dino_local_conv_rf_num_branches <= 0:
            raise ValueError("--dino_local_conv_rf_num_branches must be > 0.")
        if self.opt.dino_local_conv_rf_search_interval <= 0:
            raise ValueError("--dino_local_conv_rf_search_interval must be > 0.")
        if self.opt.dino_local_conv_rf_max_search_step < 0:
            raise ValueError("--dino_local_conv_rf_max_search_step must be >= 0.")
        if self.opt.dino_local_conv_rf_search_warmup_epochs < 0:
            raise ValueError("--dino_local_conv_rf_search_warmup_epochs must be >= 0.")
        if (self.opt.dino_lora or self.opt.dino_dora) and self.opt.dino_lora_r <= 0:
            raise ValueError("--dino_lora_r must be > 0 when LoRA or DoRA is enabled.")
        if self.opt.dino_lora_search and not self.opt.dino_lora:
            raise ValueError("--dino_lora_search requires --dino_lora to be enabled.")
        if self.opt.dino_dora and self.opt.dino_lora_search:
            raise ValueError("--dino_dora currently supports only fixed-rank baselines, not rank search.")
        if self.opt.dino_lora_search and self.opt.dino_lora_r <= 0:
            raise ValueError("--dino_lora_r must be > 0 when --dino_lora_search is enabled.")
        if self.opt.dino_lora_r_target < 0:
            raise ValueError("--dino_lora_r_target must be >= 0.")
        if self.opt.dino_lora_alpha_over_r < 0:
            raise ValueError("--dino_lora_alpha_over_r must be >= 0.")
        if self.opt.dino_lora_search_warmup_epochs < 0:
            raise ValueError("--dino_lora_search_warmup_epochs must be >= 0.")
        if self.opt.dino_lora_search_interval <= 0:
            raise ValueError("--dino_lora_search_interval must be > 0.")
        if not (0.0 <= self.opt.dino_lora_search_ema_decay < 1.0):
            raise ValueError("--dino_lora_search_ema_decay must be in [0, 1).")
        if self.opt.dino_lora_search_grad_weight < 0:
            raise ValueError("--dino_lora_search_grad_weight must be >= 0.")
        if self.opt.dino_lora_search_spectral and not self.opt.dino_lora_search:
            raise ValueError(
                "--dino_lora_search_spectral requires --dino_lora_search to be enabled."
            )
        if self.opt.dino_lora_spectral_prior_power < 0:
            raise ValueError("--dino_lora_spectral_prior_power must be >= 0.")
        if self.opt.dino_lora_spectral_uncertainty_weight < 0:
            raise ValueError(
                "--dino_lora_spectral_uncertainty_weight must be >= 0."
            )
        if self.opt.dino_lora_search_depth_buckets <= 0:
            raise ValueError("--dino_lora_search_depth_buckets must be > 0.")
        if self.opt.dino_lora_search_probe_batches < 0:
            raise ValueError("--dino_lora_search_probe_batches must be >= 0.")
        if self.opt.dino_lora_search_probe_refresh_interval < 0:
            raise ValueError("--dino_lora_search_probe_refresh_interval must be >= 0.")
        if not (0.0 < self.opt.dino_lora_search_probe_keep_ratio <= 1.0):
            raise ValueError("--dino_lora_search_probe_keep_ratio must be in (0, 1].")
        if not (0.0 < self.opt.dino_lora_search_probe_module_keep_ratio <= 1.0):
            raise ValueError(
                "--dino_lora_search_probe_module_keep_ratio must be in (0, 1]."
            )
        if self.opt.dino_lora_search_rf_delta < 0:
            raise ValueError("--dino_lora_search_rf_delta must be >= 0.")
        if self.opt.dino_lora_search_rf_temperature <= 0:
            raise ValueError("--dino_lora_search_rf_temperature must be > 0.")
        if self.opt.dino_lora_search_counterfactual and not self.opt.dino_lora_search:
            raise ValueError(
                "--dino_lora_search_counterfactual requires --dino_lora_search to be enabled."
            )
        if self.opt.dino_lora_search_counterfactual_val_batches < 0:
            raise ValueError(
                "--dino_lora_search_counterfactual_val_batches must be >= 0."
            )
        if self.opt.dino_lora_search_counterfactual_max_candidates <= 0:
            raise ValueError(
                "--dino_lora_search_counterfactual_max_candidates must be > 0."
            )
        if self.opt.dino_lora_search_counterfactual_patience <= 0:
            raise ValueError(
                "--dino_lora_search_counterfactual_patience must be > 0."
            )
        if self.opt.mfce_mid_dim <= 0:
            raise ValueError("--mfce_mid_dim must be > 0.")
        if not self.opt.mfce_aspp_rates:
            raise ValueError("--mfce_aspp_rates expects at least one dilation rate.")
        if any(rate <= 0 for rate in self.opt.mfce_aspp_rates):
            raise ValueError("--mfce_aspp_rates must use positive integers.")
        if self.opt.mfce_rf_enable and not self.opt.mfce_enable:
            raise ValueError("--mfce_rf_enable requires --mfce_enable to be enabled.")
        if self.opt.mfce_rf_num_branches <= 0:
            raise ValueError("--mfce_rf_num_branches must be > 0.")
        if (
            self.opt.mfce_rf_mode in {"rfsearch", "rfmultiple"}
            and self.opt.mfce_rf_num_branches < 2
        ):
            raise ValueError(
                "--mfce_rf_num_branches must be >= 2 for rfsearch or rfmultiple mode."
            )
        if self.opt.mfce_rf_expand_rate <= 0:
            raise ValueError("--mfce_rf_expand_rate must be > 0.")
        if self.opt.mfce_rf_min_dilation <= 0:
            raise ValueError("--mfce_rf_min_dilation must be > 0.")
        if self.opt.mfce_rf_search_interval <= 0:
            raise ValueError("--mfce_rf_search_interval must be > 0.")
        if self.opt.mfce_rf_max_search_step < 0:
            raise ValueError("--mfce_rf_max_search_step must be >= 0.")
        if self.opt.mfce_rf_init_weight < 0:
            raise ValueError("--mfce_rf_init_weight must be >= 0.")
        if self.opt.mfce_rf_search_warmup_epochs < 0:
            raise ValueError("--mfce_rf_search_warmup_epochs must be >= 0.")
        if self.opt.mfce_rf_search_epochs < 0:
            raise ValueError("--mfce_rf_search_epochs must be >= 0.")
        if self.opt.mfce_rf_diversity_weight < 0:
            raise ValueError("--mfce_rf_diversity_weight must be >= 0.")
        if self.opt.mfce_rf_diversity_margin < 0:
            raise ValueError("--mfce_rf_diversity_margin must be >= 0.")
        if self.opt.mfce_rf_log_interval < 0:
            raise ValueError("--mfce_rf_log_interval must be >= 0.")
        if self.opt.decoder_rf_num_branches <= 0:
            raise ValueError("--decoder_rf_num_branches must be > 0.")
        if (
            self.opt.decoder_rf_mode in {"rfsearch", "rfmultiple"}
            and self.opt.decoder_rf_num_branches < 2
        ):
            raise ValueError(
                "--decoder_rf_num_branches must be >= 2 for rfsearch or rfmultiple mode."
            )
        if self.opt.decoder_rf_expand_rate <= 0:
            raise ValueError("--decoder_rf_expand_rate must be > 0.")
        if self.opt.decoder_rf_min_dilation <= 0:
            raise ValueError("--decoder_rf_min_dilation must be > 0.")
        if self.opt.decoder_rf_search_interval <= 0:
            raise ValueError("--decoder_rf_search_interval must be > 0.")
        if self.opt.decoder_rf_max_search_step < 0:
            raise ValueError("--decoder_rf_max_search_step must be >= 0.")
        if self.opt.decoder_rf_init_weight < 0:
            raise ValueError("--decoder_rf_init_weight must be >= 0.")
        if self.opt.decoder_rf_search_warmup_epochs < 0:
            raise ValueError("--decoder_rf_search_warmup_epochs must be >= 0.")
        if self.opt.decoder_rf_search_epochs < 0:
            raise ValueError("--decoder_rf_search_epochs must be >= 0.")
        if self.opt.decoder_rf_log_interval < 0:
            raise ValueError("--decoder_rf_log_interval must be >= 0.")
        if self.opt.decoder_pred_guided_mode not in {
            "prob",
            "prob_uncertainty",
            "prob_uncertainty_boundary",
        }:
            raise ValueError(
                "--decoder_pred_guided_mode must be one of prob, prob_uncertainty, or prob_uncertainty_boundary."
            )
        if not (0.0 <= self.opt.dino_temporal_exchange_thresh <= 1.0):
            raise ValueError("--dino_temporal_exchange_thresh must be in [0, 1].")
        if self.opt.dino_temporal_exchange_p <= 0:
            raise ValueError("--dino_temporal_exchange_p must be > 0.")
        if not self.opt.dino_temporal_exchange_layers:
            raise ValueError("--dino_temporal_exchange_layers expects at least one index.")
        if self.opt.pairlocal_rf_enable and not self.opt.pairlocal_enable:
            raise ValueError("--pairlocal_rf_enable requires --pairlocal_enable to be enabled.")
        if self.opt.pairlocal_rf_num_branches <= 0:
            raise ValueError("--pairlocal_rf_num_branches must be > 0.")
        if (
            self.opt.pairlocal_rf_mode in {"rfsearch", "rfmultiple"}
            and self.opt.pairlocal_rf_num_branches < 2
        ):
            raise ValueError(
                "--pairlocal_rf_num_branches must be >= 2 for rfsearch or rfmultiple mode."
            )
        if self.opt.pairlocal_rf_expand_rate <= 0:
            raise ValueError("--pairlocal_rf_expand_rate must be > 0.")
        if self.opt.pairlocal_rf_min_dilation <= 0:
            raise ValueError("--pairlocal_rf_min_dilation must be > 0.")
        if self.opt.pairlocal_rf_search_interval <= 0:
            raise ValueError("--pairlocal_rf_search_interval must be > 0.")
        if self.opt.pairlocal_rf_max_search_step < 0:
            raise ValueError("--pairlocal_rf_max_search_step must be >= 0.")
        if self.opt.pairlocal_rf_init_weight < 0:
            raise ValueError("--pairlocal_rf_init_weight must be >= 0.")
        if self.opt.pairlocal_rf_search_warmup_epochs < 0:
            raise ValueError("--pairlocal_rf_search_warmup_epochs must be >= 0.")
        if self.opt.pairlocal_rf_search_epochs < 0:
            raise ValueError("--pairlocal_rf_search_epochs must be >= 0.")
        if self.opt.pairlocal_rf_log_interval < 0:
            raise ValueError("--pairlocal_rf_log_interval must be >= 0.")
        if self.opt.acpc_hidden_ratio <= 0:
            raise ValueError("--acpc_hidden_ratio must be > 0.")
        if self.opt.acpc_norm_groups <= 0:
            raise ValueError("--acpc_norm_groups must be > 0.")
        if self.opt.acpc_residual_scale < 0:
            raise ValueError("--acpc_residual_scale must be >= 0.")

        valid_search_groups = {"attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"}
        parsed_group_weights = {}
        for item in self.opt.dino_lora_search_group_weights:
            if "=" not in item:
                raise ValueError(
                    "--dino_lora_search_group_weights entries must be formatted as group=value."
                )
            group_name, value = item.split("=", 1)
            group_name = group_name.strip()
            if not any(
                group_name == base_group
                or group_name.startswith(base_group + ".")
                for base_group in valid_search_groups
            ):
                raise ValueError(
                    f"Unsupported searchable LoRA group '{group_name}'. "
                    f"Expected one of {sorted(valid_search_groups)} or a depth-bucketed variant."
                )
            try:
                weight = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid searchable LoRA group weight '{item}'."
                ) from exc
            if weight < 0:
                raise ValueError(
                    "--dino_lora_search_group_weights must use non-negative weights."
                )
            parsed_group_weights[group_name] = weight
        for group_name in valid_search_groups:
            parsed_group_weights.setdefault(group_name, 1.0)
        self.opt.dino_lora_search_group_weights = parsed_group_weights
        self.opt.dino_local_conv_blocks = sorted(set(self.opt.dino_local_conv_blocks))
        if self.opt.dino_local_conv_rf_max_dilations is not None:
            if any(rate <= 0 for rate in self.opt.dino_local_conv_rf_max_dilations):
                raise ValueError("--dino_local_conv_rf_max_dilations must use positive integers.")
            if len(self.opt.dino_local_conv_rf_max_dilations) == 1:
                self.opt.dino_local_conv_rf_max_dilations = (
                    self.opt.dino_local_conv_rf_max_dilations * len(self.opt.dino_local_conv_blocks)
                )
            elif len(self.opt.dino_local_conv_rf_max_dilations) != len(self.opt.dino_local_conv_blocks):
                raise ValueError(
                    "--dino_local_conv_rf_max_dilations expects one value or one per selected DINO local-conv block."
                )

        rf_max_dilations = self.opt.mfce_rf_max_dilations
        if rf_max_dilations is None:
            rf_max_dilations = [
                max(
                    rate + 2,
                    rate * 2 if rate <= 4 else int(round(rate * 1.5)),
                )
                for rate in self.opt.mfce_aspp_rates
            ]
        elif len(rf_max_dilations) == 1:
            rf_max_dilations = rf_max_dilations * len(self.opt.mfce_aspp_rates)
        elif len(rf_max_dilations) != len(self.opt.mfce_aspp_rates):
            raise ValueError(
                "--mfce_rf_max_dilations expects either one value or one per ASPP branch."
            )
        for seed_rate, max_rate in zip(self.opt.mfce_aspp_rates, rf_max_dilations):
            if max_rate < self.opt.mfce_rf_min_dilation:
                raise ValueError(
                    "--mfce_rf_max_dilations must be >= --mfce_rf_min_dilation."
                )
            if max_rate < seed_rate:
                raise ValueError(
                    "--mfce_rf_max_dilations must be >= the corresponding --mfce_aspp_rates seed."
                )
        self.opt.mfce_rf_max_dilations = rf_max_dilations

        decoder_rf_max_dilations = self.opt.decoder_rf_max_dilations
        if decoder_rf_max_dilations is None:
            decoder_rf_max_dilations = [6, 4, 3]
        elif len(decoder_rf_max_dilations) == 1:
            decoder_rf_max_dilations = decoder_rf_max_dilations * 3
        elif len(decoder_rf_max_dilations) != 3:
            raise ValueError(
                "--decoder_rf_max_dilations expects either one value or one per FuseGated stage."
            )
        for max_rate in decoder_rf_max_dilations:
            if max_rate < self.opt.decoder_rf_min_dilation:
                raise ValueError(
                    "--decoder_rf_max_dilations must be >= --decoder_rf_min_dilation."
                )
            if max_rate < 1:
                raise ValueError("--decoder_rf_max_dilations must be >= 1.")
        self.opt.decoder_rf_max_dilations = decoder_rf_max_dilations

        pairlocal_rf_max_dilations = self.opt.pairlocal_rf_max_dilations
        if pairlocal_rf_max_dilations is None:
            pairlocal_rf_max_dilations = [4, 4, 3, 3]
        elif len(pairlocal_rf_max_dilations) == 1:
            pairlocal_rf_max_dilations = pairlocal_rf_max_dilations * 4
        elif len(pairlocal_rf_max_dilations) != 4:
            raise ValueError(
                "--pairlocal_rf_max_dilations expects either one value or one per p2-p5 stage."
            )
        for max_rate in pairlocal_rf_max_dilations:
            if max_rate < self.opt.pairlocal_rf_min_dilation:
                raise ValueError(
                    "--pairlocal_rf_max_dilations must be >= --pairlocal_rf_min_dilation."
                )
            if max_rate < 1:
                raise ValueError("--pairlocal_rf_max_dilations must be >= 1.")
        self.opt.pairlocal_rf_max_dilations = pairlocal_rf_max_dilations

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            if self.opt.distributed:
                device_index = self.opt.gpu_ids[self.opt.local_rank]
            else:
                device_index = self.opt.gpu_ids[0]
            torch.cuda.set_device(device_index)
            self.opt.device_id = device_index
        else:
            self.opt.device_id = -1

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt
