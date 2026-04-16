import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm

from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
from option import Options
from util.metric_tool import ConfuseMatrixMeter, cm2score
from util.util import de_norm, make_numpy_grid


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def init_distributed(opt):
    if opt.distributed and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=opt.rank,
            world_size=opt.world_size,
        )


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor_sum(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_confusion_matrix(confusion_matrix, device):
    tensor = torch.tensor(confusion_matrix, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


class Trainval(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            f"cuda:{opt.device_id}" if torch.cuda.is_available() else "cpu"
        )

        self.train_loader = DataLoader(opt)
        self.train_data = self.train_loader.load_data()
        train_size = len(self.train_loader)
        if self.opt.is_main_process:
            print("#training images = %d" % train_size)

        opt.phase = "val"
        self.val_loader = DataLoader(opt)
        self.val_data = self.val_loader.load_data()
        val_size = len(self.val_loader)
        if self.opt.is_main_process:
            print("#validation images = %d" % val_size)
        opt.phase = "train"

        self.model = create_model(opt)
        self.model.configure_rf_search(len(self.train_data))
        self.lora_search_val_batches = self._build_lora_search_val_batches()
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.alpha = 0.5

        self.log_path = os.path.join(self.model.save_dir, "record.txt")
        self.vis_path = os.path.join(self.model.save_dir, opt.vis_path)
        os.makedirs(self.vis_path, exist_ok=True)

        if self.opt.is_main_process and not os.path.exists(self.log_path):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("# Record of training/validation metrics\n")
                f.write(
                    "# name: %s | backbone: %s\n"
                    % (opt.name, getattr(opt, "backbone", "NA"))
                )
                f.write(
                    "# time,epoch,train_loss,train_focal,train_dice,train_rf_div,lr,"
                )
                f.write("val_metrics(json)\n")

    def _clone_batch(self, batch):
        cloned = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        return cloned

    def _build_lora_search_val_batches(self):
        if not (
            getattr(self.opt, "dino_lora_search", False)
            and getattr(self.opt, "dino_lora_search_counterfactual", False)
        ):
            return []
        if self.opt.distributed and not self.opt.is_main_process:
            return []

        max_batches = max(
            0,
            int(getattr(self.opt, "dino_lora_search_counterfactual_val_batches", 0)),
        )
        if max_batches <= 0:
            return []

        cached_batches = []
        for batch in self.val_data:
            cached_batches.append(self._clone_batch(batch))
            if len(cached_batches) >= max_batches:
                break

        if self.opt.is_main_process:
            print(
                f"cached {len(cached_batches)} val batches for LoRA counterfactual rank search"
            )
        return cached_batches

    def _rescheduler(self, opt):
        self.model.optimizer = optim.AdamW(
            [p for p in self.model.model.parameters() if p.requires_grad],
            lr=opt.lr * 0.2,
            weight_decay=opt.weight_decay,
        )
        self.model.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.model.optimizer, int(opt.num_epochs * 0.1), eta_min=1e-7
        )
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

    def _append_log_line(self, epoch: int, train_stats: dict, val_scores: dict):
        if not self.opt.is_main_process:
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"{ts},{epoch},"
            f"{train_stats.get('loss', float('nan')):.6f},"
            f"{train_stats.get('focal', float('nan')):.6f},"
            f"{train_stats.get('dice', float('nan')):.6f},"
            f"{train_stats.get('rf_diversity', float('nan')):.6f},"
            f"{train_stats.get('lr', float('nan')):.8f},"
            + json.dumps(val_scores, ensure_ascii=False)
            + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _plot_cd_result(self, x1, x2, pred, target, epoch, stage):
        if not self.opt.is_main_process:
            return
        vis_interval = getattr(self.opt, "vis_interval", 1)
        if vis_interval <= 0 or epoch % vis_interval != 0:
            return

        if len(pred.shape) == 4:
            pred = torch.argmax(pred, dim=1)
        vis_input = make_numpy_grid(de_norm(x1[0:8]))
        vis_input2 = make_numpy_grid(de_norm(x2[0:8]))
        vis_pred = make_numpy_grid(pred[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis_gt = make_numpy_grid(target[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(self.vis_path, f"{stage}_" + str(epoch) + ".jpg")
        plt.imsave(file_name, vis)

    def train(self, epoch):
        if hasattr(self.train_loader, "set_epoch"):
            self.train_loader.set_epoch(epoch)
        tbar = tqdm(self.train_data, ncols=80) if self.opt.is_main_process else self.train_data
        self.opt.phase = "train"
        _loss = 0.0
        _focal_loss = 0.0
        _dice_loss = 0.0
        _rf_diversity = 0.0
        last_lr = self.optimizer.param_groups[0]["lr"]
        accum_steps = max(1, self.opt.grad_accum_steps)
        self.optimizer.zero_grad()
        num_batches = len(self.train_data)

        for i, data in enumerate(tbar):
            self.model.model.train()
            should_step = ((i + 1) % accum_steps == 0) or (i == num_batches - 1)

            sync_context = nullcontext()
            if (
                self.opt.distributed
                and hasattr(self.model.model, "no_sync")
                and not should_step
            ):
                sync_context = self.model.model.no_sync()

            with sync_context:
                pred, focal, dice = self.model(
                    data["img1"].to(self.device, non_blocking=True),
                    data["img2"].to(self.device, non_blocking=True),
                    data["cd_label"].to(self.device, non_blocking=True),
                )

                loss = focal * self.alpha + dice
                (loss / accum_steps).backward()

            if should_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

            _loss += loss.item()
            _focal_loss += focal.item()
            _dice_loss += dice.item()
            _rf_diversity += float(
                self.model.last_aux_losses.get("rf_diversity", 0.0)
            )
            last_lr = self.optimizer.param_groups[0]["lr"]

            if self.opt.is_main_process:
                tbar.set_description(
                    "Loss: %.3f, Focal: %.3f, Dice: %.3f, RFDiv: %.3f, LR: %.6f"
                    % (
                        _loss / (i + 1),
                        _focal_loss / (i + 1),
                        _dice_loss / (i + 1),
                        _rf_diversity / (i + 1),
                        last_lr,
                    )
                )

            if i == num_batches - 1:
                self._plot_cd_result(
                    data["img1"], data["img2"], pred, data["cd_label"], epoch, "train"
                )

        self.schedular.step()

        batch_count = max(1, i + 1)
        reduced = reduce_tensor_sum(
            [_loss, _focal_loss, _dice_loss, _rf_diversity, batch_count],
            self.device,
        )
        denom = reduced[4].item()
        return {
            "loss": float(reduced[0].item() / denom),
            "focal": float(reduced[1].item() / denom),
            "dice": float(reduced[2].item() / denom),
            "rf_diversity": float(reduced[3].item() / denom),
            "lr": last_lr,
        }

    def val(self, epoch):
        tbar = tqdm(self.val_data, ncols=80) if self.opt.is_main_process else self.val_data
        self.running_metric.clear()
        self.opt.phase = "val"
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(tbar):
                val_pred = self.model.inference(
                    data["img1"].to(self.device, non_blocking=True),
                    data["img2"].to(self.device, non_blocking=True),
                )
                val_target = data["cd_label"].detach()
                val_pred = torch.argmax(val_pred.detach(), dim=1)
                _ = self.running_metric.update_cm(
                    pr=val_pred.cpu().numpy(),
                    gt=val_target.cpu().numpy(),
                )
                if i == len(self.val_data) - 1:
                    self._plot_cd_result(
                        data["img1"],
                        data["img2"],
                        val_pred,
                        data["cd_label"],
                        epoch,
                        "val",
                    )

        local_confusion = (
            self.running_metric.sum
            if self.running_metric.initialized
            else np.zeros((2, 2), dtype=np.float64)
        )
        global_confusion = reduce_confusion_matrix(local_confusion, self.device)
        val_scores = cm2score(global_confusion)

        if self.opt.is_main_process:
            message = "(phase: %s) " % (self.opt.phase)
            for k, v in val_scores.items():
                message += "%s: %.3f " % (k, v * 100)
            print(message)

        if self.opt.is_main_process and val_scores.get("iou_1", 0.0) >= self.previous_best:
            self.model.save(self.opt.name, self.opt.backbone)
            self.previous_best = val_scores["iou_1"]

        return val_scores


if __name__ == "__main__":
    opt = Options().parse()
    init_distributed(opt)
    setup_seed(seed=1 + opt.rank)

    try:
        trainval = Trainval(opt)

        for epoch in range(1, opt.num_epochs + 1):
            if opt.is_main_process:
                print(
                    "\n==> Name %s, Epoch %i, previous best = %.3f"
                    % (opt.name, epoch, trainval.previous_best * 100)
                )
            trainval.model.update_lora_rank_search_with_val(
                epoch,
                trainval.lora_search_val_batches,
                focal_weight=trainval.alpha,
            )
            if epoch == int(opt.num_epochs * 0.9):
                trainval._rescheduler(opt)
            train_stats = trainval.train(epoch)
            val_scores = trainval.val(epoch)
            trainval._append_log_line(epoch, train_stats, val_scores)
            if (
                opt.is_main_process
                and opt.mfce_rf_enable
                and opt.mfce_rf_log_interval > 0
                and epoch % opt.mfce_rf_log_interval == 0
            ):
                trainval.model._log_rf_states(f"epoch {epoch}")

        if opt.is_main_process:
            print("Done!")
    finally:
        cleanup_distributed()
