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
        if self.opt.acpc_hidden_ratio <= 0:
            raise ValueError("--acpc_hidden_ratio must be > 0.")
        if self.opt.acpc_norm_groups <= 0:
            raise ValueError("--acpc_norm_groups must be > 0.")
        if self.opt.acpc_residual_scale < 0:
            raise ValueError("--acpc_residual_scale must be >= 0.")

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
