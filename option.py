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
            help="per-group budget weights for searchable LoRA, formatted as group=value",
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
        if self.opt.dino_lora_search and not self.opt.dino_lora:
            raise ValueError("--dino_lora_search requires --dino_lora to be enabled.")
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
        if self.opt.mfce_mid_dim <= 0:
            raise ValueError("--mfce_mid_dim must be > 0.")
        if not self.opt.mfce_aspp_rates:
            raise ValueError("--mfce_aspp_rates expects at least one dilation rate.")
        if any(rate <= 0 for rate in self.opt.mfce_aspp_rates):
            raise ValueError("--mfce_aspp_rates must use positive integers.")
        if not (0.0 <= self.opt.dino_temporal_exchange_thresh <= 1.0):
            raise ValueError("--dino_temporal_exchange_thresh must be in [0, 1].")
        if self.opt.dino_temporal_exchange_p <= 0:
            raise ValueError("--dino_temporal_exchange_p must be > 0.")
        if not self.opt.dino_temporal_exchange_layers:
            raise ValueError("--dino_temporal_exchange_layers expects at least one index.")
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
            if group_name not in valid_search_groups:
                raise ValueError(
                    f"Unsupported searchable LoRA group '{group_name}'. "
                    f"Expected one of {sorted(valid_search_groups)}."
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
