import os

import torch
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .ChangeDINO_crossgate import ChangeModel
from .loss.dice import DICELoss
from .loss.focal import FocalLoss


def get_model(backbone_name="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1], **kwargs):
    return ChangeModel(backbone_name, fpn_channels, n_layers=n_layers, **kwargs)


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
            dino_lora_r=opt.dino_lora_r,
            dino_lora_alpha=opt.dino_lora_alpha,
            dino_lora_dropout=opt.dino_lora_dropout,
            dino_lora_search=opt.dino_lora_search,
            dino_lora_r_target=opt.dino_lora_r_target,
            dino_lora_alpha_over_r=opt.dino_lora_alpha_over_r,
            dino_lora_search_warmup_epochs=opt.dino_lora_search_warmup_epochs,
            dino_lora_search_interval=opt.dino_lora_search_interval,
            crossgate_attn_dim=opt.crossgate_attn_dim,
            crossgate_num_heads=opt.crossgate_num_heads,
            crossgate_window_size=opt.crossgate_window_size,
            crossgate_gamma_init=opt.crossgate_gamma_init,
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

        if self.opt.dino_lora:
            dino_trainable = [
                (name, numel)
                for name, numel in trainable
                if name.startswith("encoder.dino.model.")
                or name.startswith("module.encoder.dino.model.")
            ]
            lora_trainable = [
                (name, numel)
                for name, numel in trainable
                if ".lora_" in name
            ]
            print(f"trainable DINO tensors: {len(dino_trainable)}")
            print(
                f"trainable DINO parameters total: {sum(numel for _, numel in dino_trainable)}"
            )
            print(f"trainable LoRA tensors: {len(lora_trainable)}")
            print(
                f"trainable LoRA parameters total: {sum(numel for _, numel in lora_trainable)}"
            )

            if not lora_trainable:
                raise RuntimeError(
                    "LoRA is enabled, but no trainable LoRA parameters were found."
                )

            preview_limit = 12
            print("sample trainable LoRA parameter names:")
            for name, _ in lora_trainable[:preview_limit]:
                print(f"  {name}")

    def update_lora_rank_search(self, epoch):
        if not getattr(self.opt, "dino_lora_search", False):
            return None

        network = self._unwrap_model(self.model)
        if not hasattr(network.encoder.dino, "update_lora_rank_search"):
            return None

        summary = network.encoder.dino.update_lora_rank_search(epoch, self.opt.num_epochs)
        if summary and self.opt.is_main_process:
            preview = ", ".join(str(rank) for rank in summary["active_ranks"][:8])
            suffix = " ..." if len(summary["active_ranks"]) > 8 else ""
            print(
                f"LoRA rank search -> budget={summary['budget_rank']}, "
                f"total_active={summary['total_active_rank']}, "
                f"layer_ranks=[{preview}{suffix}]"
            )
        return summary

    def forward(self, x1, x2, label):
        final_pred, preds = self.model(x1, x2)
        label = label.long()
        focal = self.focal(final_pred, label)
        dice = self.dice(final_pred, label)
        for i in range(len(preds)):
            focal += self.focal(preds[i], label)
            dice += 0.5 * self.dice(preds[i], label)
        return final_pred, focal, dice

    @torch.inference_mode()
    def inference(self, x1, x2):
        return self._unwrap_model(self.model)._forward(x1, x2)

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
        checkpoint = torch.load(save_path, map_location=self.device, weights_only=True)
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
