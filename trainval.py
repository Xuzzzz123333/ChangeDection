import json
import math
import os
import random
import shutil
import csv
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


class DynamicPolicyUsageMeter:
    def __init__(self, metadata, enabled, use_block_policy, use_token_policy, use_mlp_policy):
        self.enabled = bool(enabled)
        self.num_layers = int(metadata.get("num_layers", 0)) if metadata else 0
        self.num_heads = int(metadata.get("num_heads", 0)) if metadata else 0
        self.num_patch_tokens = int(metadata.get("num_patch_tokens", 0)) if metadata else 0
        self.num_mlp_chunks = int(metadata.get("num_mlp_chunks", 0)) if metadata else 0
        self.use_block_policy = bool(use_block_policy)
        self.use_token_policy = bool(use_token_policy)
        self.use_mlp_policy = bool(use_mlp_policy)
        self.reset()

    def reset(self):
        self.num_samples = 0
        self.head_sum = np.zeros((self.num_layers, self.num_heads), dtype=np.float64)
        self.head_active_sum = np.zeros((self.num_layers, self.num_heads), dtype=np.float64)
        self.block_sum = np.zeros((self.num_layers,), dtype=np.float64)
        self.token_sum = 0.0
        self.mlp_sum = np.zeros((self.num_layers, self.num_mlp_chunks), dtype=np.float64)
        self.value_sum = 0.0
        self.value_sq_sum = 0.0
        self.value_count = 0
        self.value_min = float("inf")
        self.value_max = float("-inf")
        self.token_example = None

    def update(self, policy_state):
        if not self.enabled or policy_state is None:
            return
        head_keep = policy_state.get("head_keep")
        block_keep = policy_state.get("block_keep")
        token_keep = policy_state.get("token_keep")
        mlp_keep = policy_state.get("mlp_keep")

        batch_size = 0
        if head_keep is not None:
            head_keep = head_keep.detach().float().cpu().numpy()
            batch_size = int(head_keep.shape[0])
            self.head_sum += head_keep.sum(axis=0)
            self.head_active_sum += (head_keep > 0.5).astype(np.float32).sum(axis=0)
            self.value_sum += float(head_keep.sum())
            self.value_sq_sum += float(np.square(head_keep).sum())
            self.value_count += int(head_keep.size)
            self.value_min = min(self.value_min, float(head_keep.min()))
            self.value_max = max(self.value_max, float(head_keep.max()))
        if block_keep is not None:
            block_keep = block_keep.detach().float().cpu().numpy()
            if batch_size == 0:
                batch_size = int(block_keep.shape[0])
            self.block_sum += block_keep.mean(axis=-1).sum(axis=0)
            if head_keep is None:
                self.value_sum += float(block_keep.sum())
                self.value_sq_sum += float(np.square(block_keep).sum())
                self.value_count += int(block_keep.size)
                self.value_min = min(self.value_min, float(block_keep.min()))
                self.value_max = max(self.value_max, float(block_keep.max()))
        if token_keep is not None:
            token_keep = token_keep.detach().float().cpu().numpy()
            if batch_size == 0:
                batch_size = int(token_keep.shape[0])
            if token_keep.ndim >= 3:
                self.num_patch_tokens = int(token_keep.shape[-1])
            self.token_sum += float(token_keep.sum())
            if self.token_example is None and token_keep.shape[0] > 0:
                self.token_example = token_keep[0, 0].copy()
            if head_keep is None and block_keep is None:
                self.value_sum += float(token_keep.sum())
                self.value_sq_sum += float(np.square(token_keep).sum())
                self.value_count += int(token_keep.size)
                self.value_min = min(self.value_min, float(token_keep.min()))
                self.value_max = max(self.value_max, float(token_keep.max()))
        if mlp_keep is not None:
            mlp_keep = mlp_keep.detach().float().cpu().numpy()
            if batch_size == 0:
                batch_size = int(mlp_keep.shape[0])
            self.mlp_sum += mlp_keep.sum(axis=0)
            self.value_sum += float(mlp_keep.sum())
            self.value_sq_sum += float(np.square(mlp_keep).sum())
            self.value_count += int(mlp_keep.size)
            self.value_min = min(self.value_min, float(mlp_keep.min()))
            self.value_max = max(self.value_max, float(mlp_keep.max()))
        self.num_samples += batch_size

    def summary(self):
        if not self.enabled or self.num_samples <= 0 or self.num_layers <= 0:
            return {
                "mean_head_keep": 1.0,
                "head_active_ratio": 1.0,
                "per_layer_head_keep": [1.0] * self.num_layers,
                "per_layer_head_active_ratio": [1.0] * self.num_layers,
                "head_heatmap": np.ones((self.num_layers, self.num_heads), dtype=np.float32),
                "mean_block_keep": 1.0,
                "per_layer_block_keep": [1.0] * self.num_layers,
                "mean_token_keep": 1.0,
                "mean_mlp_keep": 1.0,
                "per_layer_mlp_keep": [1.0] * self.num_layers,
                "policy_min": 1.0,
                "policy_max": 1.0,
                "policy_std": 0.0,
                "token_example": None,
            }

        if self.num_heads > 0:
            head_heatmap = (self.head_sum / max(self.num_samples, 1)).astype(np.float32)
            head_active_heatmap = (self.head_active_sum / max(self.num_samples, 1)).astype(np.float32)
            per_layer_head_keep = head_heatmap.mean(axis=1).tolist()
            per_layer_head_active_ratio = head_active_heatmap.mean(axis=1).tolist()
            mean_head_keep = float(
                self.head_sum.sum() / max(self.num_samples * self.num_layers * self.num_heads, 1)
            )
            head_active_ratio = float(
                self.head_active_sum.sum() / max(self.num_samples * self.num_layers * self.num_heads, 1)
            )
        else:
            head_heatmap = np.ones((self.num_layers, 0), dtype=np.float32)
            per_layer_head_keep = [1.0] * self.num_layers
            per_layer_head_active_ratio = [1.0] * self.num_layers
            mean_head_keep = 1.0
            head_active_ratio = 1.0

        if self.use_block_policy:
            per_layer_block_keep = (self.block_sum / max(self.num_samples, 1)).astype(np.float32).tolist()
            mean_block_keep = float(np.mean(per_layer_block_keep))
        else:
            per_layer_block_keep = [1.0] * self.num_layers
            mean_block_keep = 1.0

        if self.use_token_policy:
            denom = max(self.num_samples * self.num_layers * max(self.num_patch_tokens, 1), 1)
            mean_token_keep = float(self.token_sum / denom)
        else:
            mean_token_keep = 1.0

        if self.use_mlp_policy and self.num_mlp_chunks > 0:
            mlp_keep = (self.mlp_sum / max(self.num_samples, 1)).astype(np.float32)
            per_layer_mlp_keep = mlp_keep.mean(axis=1).tolist()
            mean_mlp_keep = float(
                self.mlp_sum.sum() / max(self.num_samples * self.num_layers * self.num_mlp_chunks, 1)
            )
        else:
            per_layer_mlp_keep = [1.0] * self.num_layers
            mean_mlp_keep = 1.0

        if self.value_count > 0:
            mean_value = self.value_sum / float(self.value_count)
            variance = max(self.value_sq_sum / float(self.value_count) - mean_value * mean_value, 0.0)
            policy_std = math.sqrt(variance)
            policy_min = float(self.value_min)
            policy_max = float(self.value_max)
        else:
            policy_std = 0.0
            policy_min = 1.0
            policy_max = 1.0

        return {
            "mean_head_keep": mean_head_keep,
            "head_active_ratio": head_active_ratio,
            "per_layer_head_keep": per_layer_head_keep,
            "per_layer_head_active_ratio": per_layer_head_active_ratio,
            "head_heatmap": head_heatmap,
            "mean_block_keep": mean_block_keep,
            "per_layer_block_keep": per_layer_block_keep,
            "mean_token_keep": mean_token_keep,
            "mean_mlp_keep": mean_mlp_keep,
            "per_layer_mlp_keep": per_layer_mlp_keep,
            "policy_min": policy_min,
            "policy_max": policy_max,
            "policy_std": policy_std,
            "token_example": self.token_example,
        }


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

        self.eval_phase = str(getattr(opt, "eval_phase", "val"))
        opt.phase = self.eval_phase
        self.val_loader = DataLoader(opt)
        self.val_data = self.val_loader.load_data()
        val_size = len(self.val_loader)
        if self.opt.is_main_process:
            print(f"#{self.eval_phase} images = {val_size}")
        opt.phase = "train"

        self.model = create_model(opt)
        self.model.configure_rf_search(len(self.train_data))
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.alpha = 0.5

        self.log_path = os.path.join(self.model.save_dir, "record.txt")
        self.checkpoint_index_path = os.path.join(
            self.model.save_dir, "checkpoint_index.json"
        )
        self.best_metrics_path = os.path.join(
            self.model.save_dir, "best_metrics.json"
        )
        self.vis_path = os.path.join(self.model.save_dir, opt.vis_path)
        self.policy_usage_csv_path = os.path.join(
            self.model.save_dir, "policy_usage_epoch.csv"
        )
        self.policy_heatmap_path = os.path.join(
            self.model.save_dir, "head_policy_heatmap_best.png"
        )
        self.policy_block_curve_path = os.path.join(
            self.model.save_dir, "block_policy_curve_best.png"
        )
        self.policy_token_example_path = os.path.join(
            self.model.save_dir, "token_policy_example.png"
        )
        os.makedirs(self.vis_path, exist_ok=True)
        self.dynamic_policy_metadata = (
            self.model.get_dynamic_policy_metadata()
            if hasattr(self.model, "get_dynamic_policy_metadata")
            else None
        )
        self.policy_usage_rows = []
        self.last_train_policy_summary = None
        self.last_val_policy_summary = None
        self.topk_checkpoints = []
        self.best_metric_records = {}
        self.last_checkpoints = []
        self.topk_metric_name = "iou_1"

        if self.opt.is_main_process and self.opt.save_multi_best:
            print(
                "save_multi_best is ignored; checkpoint saving now keeps only "
                "top-k checkpoints ranked by iou_1."
            )

        if self.opt.is_main_process and not os.path.exists(self.log_path):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("# Record of training/validation metrics\n")
                f.write(
                    "# name: %s | backbone: %s\n"
                    % (opt.name, getattr(opt, "backbone", "NA"))
                )
                f.write(
                    "# time,epoch,train_loss,train_focal,train_dice,train_rf_div,lr,"
                    "soft_gate_budget_lambda,soft_gate_budget_loss,"
                    "soft_gate_effective_rank_total,soft_gate_expected_rank_ratio_mean,"
                    "cgla_temporal_reg_lambda,cgla_temporal_reg_loss,"
                    "cgla_temporal_reg_layers,cgla_temporal_reg_response_mean,"
                    "cgla_temporal_reg_change_loss,cgla_temporal_reg_unchange_loss,"
                    "cgla_temporal_reg_mask_mean,cgla_temporal_reg_mask_nonzero_ratio,"
                )
                f.write("val_metrics(json),spectral_metrics(json)\n")
        if (
            self.opt.is_main_process
            and getattr(self.opt, "use_dynamic_policy", False)
            and not os.path.exists(self.policy_usage_csv_path)
        ):
            self._write_policy_usage_csv()

        # Build fixed representative anchor sets so search decisions are
        # comparable across epochs.
        self.lora_search_val_batches = self._build_lora_search_val_batches()
        self.lora_search_probe_batches = self._build_lora_search_probe_batches()

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

    def _extract_lora_search_feature(self, sample):
        img1 = sample["img1"].float()
        img2 = sample["img2"].float()
        label = sample["cd_label"].float()
        diff = (img1 - img2).abs()

        img1_mean = img1.mean(dim=(1, 2))
        img2_mean = img2.mean(dim=(1, 2))
        diff_mean = diff.mean(dim=(1, 2))
        diff_std = diff.std(dim=(1, 2), unbiased=False)

        change_ratio = label.mean()
        label_std = label.std(unbiased=False)
        h_edge = (
            (label[1:, :] != label[:-1, :]).float().mean()
            if label.shape[0] > 1
            else label.new_zeros(())
        )
        w_edge = (
            (label[:, 1:] != label[:, :-1]).float().mean()
            if label.shape[1] > 1
            else label.new_zeros(())
        )

        feature = torch.cat(
            [
                img1_mean,
                img2_mean,
                diff_mean,
                diff_std,
                torch.stack([change_ratio, label_std, h_edge, w_edge]),
            ],
            dim=0,
        )
        return feature.cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _kmeans_representative_indices(features: np.ndarray, num_clusters: int):
        num_samples = features.shape[0]
        if num_clusters >= num_samples:
            return list(range(num_samples))
        if num_clusters <= 1:
            center = features.mean(axis=0, keepdims=True)
            distances = ((features - center) ** 2).sum(axis=1)
            return [int(np.argmin(distances))]

        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        normalized = (features - mean) / np.clip(std, a_min=1e-6, a_max=None)

        rng = np.random.default_rng(20260418)
        center_indices = [int(np.argmax(np.linalg.norm(normalized, axis=1)))]
        while len(center_indices) < num_clusters:
            center_feats = normalized[center_indices]
            distances = ((normalized[:, None, :] - center_feats[None, :, :]) ** 2).sum(
                axis=2
            )
            min_dist = distances.min(axis=1)
            min_dist[center_indices] = 0.0
            if float(min_dist.sum()) <= 1e-12:
                remaining = [index for index in range(num_samples) if index not in center_indices]
                center_indices.append(int(remaining[0]))
                continue
            next_index = int(rng.choice(num_samples, p=min_dist / min_dist.sum()))
            if next_index in center_indices:
                remaining = [index for index in range(num_samples) if index not in center_indices]
                next_index = int(remaining[0])
            center_indices.append(next_index)

        centers = normalized[center_indices].copy()
        assignments = np.zeros(num_samples, dtype=np.int64)
        for _ in range(12):
            distances = ((normalized[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            assignments = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster_index in range(num_clusters):
                member_mask = assignments == cluster_index
                if np.any(member_mask):
                    new_centers[cluster_index] = normalized[member_mask].mean(axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers

        selected_indices = []
        used_indices = set()
        for cluster_index in range(num_clusters):
            member_indices = np.where(assignments == cluster_index)[0]
            if member_indices.size == 0:
                continue
            member_feats = normalized[member_indices]
            center = centers[cluster_index : cluster_index + 1]
            distances = ((member_feats - center) ** 2).sum(axis=1)
            ordered = member_indices[np.argsort(distances)]
            chosen_index = None
            for candidate_index in ordered.tolist():
                if candidate_index not in used_indices:
                    chosen_index = int(candidate_index)
                    break
            if chosen_index is None:
                chosen_index = int(ordered[0])
            selected_indices.append(chosen_index)
            used_indices.add(chosen_index)

        if len(selected_indices) < num_clusters:
            remaining = [index for index in range(num_samples) if index not in used_indices]
            if not selected_indices:
                selected_indices.extend(remaining[:num_clusters])
            else:
                selected_feats = normalized[selected_indices]
                while remaining and len(selected_indices) < num_clusters:
                    remaining_feats = normalized[remaining]
                    distances = (
                        (
                            remaining_feats[:, None, :]
                            - selected_feats[None, :, :]
                        )
                        ** 2
                    ).sum(axis=2)
                    next_pos = int(np.argmax(distances.min(axis=1)))
                    next_index = int(remaining[next_pos])
                    selected_indices.append(next_index)
                    used_indices.add(next_index)
                    remaining.pop(next_pos)
                    selected_feats = normalized[selected_indices]

        return selected_indices[:num_clusters]

    def _build_representative_batches(self, dataset, max_batches, label):
        if self.opt.distributed and not self.opt.is_main_process:
            return []

        max_batches = max(0, int(max_batches))
        if max_batches <= 0:
            return []

        dataset_size = len(dataset)
        if dataset_size <= 0:
            return []

        batch_size = max(1, int(self.opt.batch_size))
        total_samples = min(dataset_size, max_batches * batch_size)
        if total_samples <= 0:
            return []

        if total_samples >= dataset_size:
            sample_indices = list(range(dataset_size))
        else:
            features = []
            for index in range(dataset_size):
                features.append(self._extract_lora_search_feature(dataset[index]))
            sample_indices = self._kmeans_representative_indices(
                np.stack(features, axis=0),
                total_samples,
            )

        cached_batches = []
        for start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[start : start + batch_size]
            batch_items = [dataset[index] for index in batch_indices]
            cached_batches.append(self._clone_batch(DataLoader._collate_fn(batch_items)))

        if self.opt.is_main_process:
            print(
                f"built {len(cached_batches)} clustered representative {label} batches "
                f"({len(sample_indices)} samples)"
            )
        return cached_batches

    def _build_lora_search_val_batches(self):
        if not (
            getattr(self.opt, "dino_lora_search", False)
            and getattr(self.opt, "dino_lora_search_counterfactual", False)
        ):
            return []
        return self._build_representative_batches(
            self.val_loader.dataset,
            getattr(self.opt, "dino_lora_search_counterfactual_val_batches", 0),
            "val anchors for LoRA counterfactual rank search",
        )

    def _build_lora_search_probe_batches(self):
        if not (
            getattr(self.opt, "dino_lora_search", False)
            and getattr(self.opt, "dino_lora_search_strategy", "classic") == "rfnext"
        ):
            return []
        return self._build_representative_batches(
            self.train_loader.dataset,
            getattr(self.opt, "dino_lora_search_probe_batches", 0),
            "train probe batches for RF-style LoRA search",
        )

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

    @staticmethod
    def _safe_metric(scores, key):
        try:
            value = float(scores.get(key, float("-inf")))
        except (TypeError, ValueError, AttributeError):
            return float("-inf")
        return value if math.isfinite(value) else float("-inf")

    @staticmethod
    def _normalize_scores(scores):
        normalized = {}
        for key, value in scores.items():
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                normalized[key] = value
        return normalized

    @staticmethod
    def _sanitize_metric_name(metric_name):
        return "".join(ch if ch.isalnum() else "_" for ch in metric_name)

    def _format_checkpoint_name(self, metric_name, epoch, scores):
        metric_tag = self._sanitize_metric_name(metric_name)
        epoch_width = max(3, len(str(max(1, int(self.opt.num_epochs)))))
        parts = [f"best_{metric_tag}", f"epoch{int(epoch):0{epoch_width}d}"]
        summary_keys = []
        for key in (metric_name, "iou_1", "F1_1", "miou"):
            if key in scores and key not in summary_keys:
                summary_keys.append(key)
        for key in summary_keys:
            value = self._safe_metric(scores, key)
            if math.isfinite(value):
                parts.append(f"{self._sanitize_metric_name(key)}_{value:.4f}")
        return "_".join(parts) + ".pth"

    def _save_checkpoint_path(self, path, epoch, scores, extra=None):
        normalized_scores = self._normalize_scores(scores)
        checkpoint_extra = dict(extra or {})
        if getattr(self.opt, "use_dynamic_policy", False) and self.last_val_policy_summary is not None:
            checkpoint_extra["dynamic_policy_summary"] = self._checkpointable_policy_summary(
                self.last_val_policy_summary
            )
        self.model.save_checkpoint(
            path,
            epoch=int(epoch),
            scores=normalized_scores,
            extra=checkpoint_extra,
        )
        metric_name = checkpoint_extra.get("metric_name", "metric")
        metric_value = float(checkpoint_extra.get("metric_value", float("nan")))
        print(
            f"Saved checkpoint: {path}, metric_name={metric_name}, "
            f"metric_value={metric_value:.6f}, epoch={int(epoch)}"
        )

    @staticmethod
    def _checkpointable_policy_summary(summary):
        if summary is None:
            return None
        payload = {}
        for key, value in summary.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                payload[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in value
                ]
            else:
                payload[key] = value
        return payload

    def _cleanup_multi_best_artifacts(self):
        metric_names = set(getattr(self.opt, "save_best_metrics", []) or [])
        metric_names.add(getattr(self.opt, "save_top_k_metric", "iou_1"))
        metric_names.add("iou_1")
        for metric_name in metric_names:
            alias_path = os.path.join(
                self.model.save_dir,
                f"best_{self._sanitize_metric_name(metric_name)}.pth",
            )
            if os.path.exists(alias_path):
                os.remove(alias_path)
        if os.path.exists(self.best_metrics_path):
            os.remove(self.best_metrics_path)

    def _write_checkpoint_index(self):
        if not self.opt.is_main_process:
            return
        payload = {
            "save_top_k": int(self.opt.save_top_k),
            "save_top_k_metric": self.topk_metric_name,
            "topk_checkpoints": self.topk_checkpoints,
            "best_metric_records": self.best_metric_records,
            "last_checkpoints": self.last_checkpoints,
        }
        with open(self.checkpoint_index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _update_topk_checkpoints(self, epoch, val_scores):
        metric_name = self.topk_metric_name
        metric_value = self._safe_metric(val_scores, metric_name)
        if not math.isfinite(metric_value):
            return

        top_k = int(self.opt.save_top_k)
        if len(self.topk_checkpoints) >= top_k:
            worst_metric = min(item["metric"] for item in self.topk_checkpoints)
            if metric_value <= worst_metric:
                return

        ckpt_path = os.path.join(
            self.model.save_dir,
            self._format_checkpoint_name(metric_name, epoch, val_scores),
        )
        extra = {
            "kind": "topk",
            "metric_name": metric_name,
            "metric_value": metric_value,
        }
        self._save_checkpoint_path(ckpt_path, epoch, val_scores, extra=extra)
        record = {
            "epoch": int(epoch),
            "metric": float(metric_value),
            "path": ckpt_path,
            "scores": self._normalize_scores(val_scores),
        }
        self.topk_checkpoints.append(record)
        self.topk_checkpoints.sort(
            key=lambda item: (item["metric"], item["epoch"]),
            reverse=True,
        )

        while len(self.topk_checkpoints) > top_k:
            removed = self.topk_checkpoints.pop(-1)
            removed_path = removed.get("path")
            if removed_path and os.path.exists(removed_path):
                os.remove(removed_path)

        if not self.topk_checkpoints:
            return

        best_record = self.topk_checkpoints[0]
        legacy_alias = os.path.join(
            self.model.save_dir,
            f"{self.opt.name}_{self.opt.backbone}_best.pth",
        )
        if best_record["path"] != legacy_alias:
            shutil.copyfile(best_record["path"], legacy_alias)
        self.best_metric_records = {}
        self._cleanup_multi_best_artifacts()

    def _save_periodic_last(self, epoch, val_scores):
        interval = int(self.opt.save_last_every)
        if interval <= 0 or epoch % interval != 0:
            return

        ckpt_path = os.path.join(self.model.save_dir, f"last_epoch{int(epoch):03d}.pth")
        extra = {
            "kind": "last",
            "metric_name": "epoch",
            "metric_value": float(epoch),
        }
        self._save_checkpoint_path(ckpt_path, epoch, val_scores, extra=extra)
        self.last_checkpoints.append(
            {
                "epoch": int(epoch),
                "metric": float(epoch),
                "path": ckpt_path,
                "scores": self._normalize_scores(val_scores),
            }
        )
        self.last_checkpoints.sort(key=lambda item: item["epoch"], reverse=True)
        while len(self.last_checkpoints) > 2:
            removed = self.last_checkpoints.pop(-1)
            removed_path = removed.get("path")
            if removed_path and os.path.exists(removed_path):
                os.remove(removed_path)

    def _append_log_line(self, epoch: int, train_stats: dict, val_scores: dict):
        if not self.opt.is_main_process:
            return

        spectral_metrics = {}
        if getattr(self.opt, "dino_lora_search_spectral", False):
            spectral_metrics = {
                "search": getattr(self.model, "last_spectral_search_stats", {}),
                "probe": getattr(self.model, "last_spectral_probe_stats", {}),
                "first_jump": train_stats.get("spectral_first_search_jump", {}),
            }
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"{ts},{epoch},"
            f"{train_stats.get('loss', float('nan')):.6f},"
            f"{train_stats.get('focal', float('nan')):.6f},"
            f"{train_stats.get('dice', float('nan')):.6f},"
            f"{train_stats.get('rf_diversity', float('nan')):.6f},"
            f"{train_stats.get('lr', float('nan')):.8f},"
            f"{train_stats.get('soft_gate_budget_lambda', 0.0):.6f},"
            f"{train_stats.get('soft_gate_budget_loss', 0.0):.6f},"
            f"{train_stats.get('soft_gate_effective_rank_total', 0.0):.6f},"
            f"{train_stats.get('soft_gate_expected_rank_ratio_mean', 0.0):.6f},"
            f"{train_stats.get('cgla_temporal_reg_lambda', 0.0):.6f},"
            f"{train_stats.get('cgla_temporal_reg_loss', 0.0):.6f},"
            f"{train_stats.get('cgla_temporal_reg_layers', 0.0):.6f},"
            f"{train_stats.get('cgla_temporal_reg_response_mean', 0.0):.6f},"
            f"{train_stats.get('cgla_temporal_reg_change_loss', 0.0):.6f},"
            f"{train_stats.get('cgla_temporal_reg_unchange_loss', 0.0):.6f},"
            f"{train_stats.get('cgla_temporal_reg_mask_mean', 0.0):.6f},"
            f"{train_stats.get('cgla_temporal_reg_mask_nonzero_ratio', 0.0):.6f},"
            + json.dumps(val_scores, ensure_ascii=False)
            + ","
            + json.dumps(spectral_metrics, ensure_ascii=False)
            + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _new_policy_meter(self):
        return DynamicPolicyUsageMeter(
            metadata=self.dynamic_policy_metadata or {},
            enabled=getattr(self.opt, "use_dynamic_policy", False),
            use_block_policy=getattr(self.opt, "use_block_policy", False),
            use_token_policy=getattr(self.opt, "use_token_policy", False),
            use_mlp_policy=getattr(self.opt, "use_mlp_policy", False),
        )

    def _write_policy_usage_csv(self):
        if not self.opt.is_main_process:
            return
        num_layers = int((self.dynamic_policy_metadata or {}).get("num_layers", 0))
        fieldnames = [
            "epoch",
            "split",
            "mean_head_keep",
            "head_active_ratio",
            "mean_block_keep",
            "mean_token_keep",
            "mean_mlp_keep",
            "target_compute_ratio",
            "policy_cost",
            "policy_budget_loss",
            "policy_min",
            "policy_max",
            "policy_std",
        ]
        fieldnames += [f"layer_{idx}_head_keep" for idx in range(num_layers)]
        fieldnames += [f"layer_{idx}_head_active" for idx in range(num_layers)]
        fieldnames += [f"layer_{idx}_mlp_keep" for idx in range(num_layers)]
        with open(self.policy_usage_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.policy_usage_rows:
                writer.writerow(row)

    def _record_policy_usage(self, epoch, split, summary, aux_stats):
        if not self.opt.is_main_process or not getattr(self.opt, "use_dynamic_policy", False):
            return
        policy_cost = (
            float(getattr(self.opt, "policy_head_weight", 1.0))
            * float(summary.get("mean_head_keep", 1.0))
            + float(getattr(self.opt, "policy_block_weight", 0.0))
            * float(summary.get("mean_block_keep", 1.0))
            + float(getattr(self.opt, "policy_token_weight", 0.0))
            * float(summary.get("mean_token_keep", 1.0))
            + float(getattr(self.opt, "policy_mlp_weight", 0.0))
            * float(summary.get("mean_mlp_keep", 1.0))
        )
        row = {
            "epoch": int(epoch),
            "split": str(split),
            "mean_head_keep": float(summary.get("mean_head_keep", 1.0)),
            "head_active_ratio": float(summary.get("head_active_ratio", 1.0)),
            "mean_block_keep": float(summary.get("mean_block_keep", 1.0)),
            "mean_token_keep": float(summary.get("mean_token_keep", 1.0)),
            "mean_mlp_keep": float(summary.get("mean_mlp_keep", 1.0)),
            "target_compute_ratio": float(
                aux_stats.get("dynamic_policy_target_compute_ratio", 1.0)
            ),
            "policy_cost": float(policy_cost),
            "policy_budget_loss": float(
                abs(
                    policy_cost
                    - float(aux_stats.get("dynamic_policy_target_compute_ratio", 1.0))
                )
            ),
            "policy_min": float(summary.get("policy_min", 1.0)),
            "policy_max": float(summary.get("policy_max", 1.0)),
            "policy_std": float(summary.get("policy_std", 0.0)),
        }
        for idx, value in enumerate(summary.get("per_layer_head_keep", [])):
            row[f"layer_{idx}_head_keep"] = float(value)
        for idx, value in enumerate(summary.get("per_layer_head_active_ratio", [])):
            row[f"layer_{idx}_head_active"] = float(value)
        for idx, value in enumerate(summary.get("per_layer_mlp_keep", [])):
            row[f"layer_{idx}_mlp_keep"] = float(value)
        self.policy_usage_rows.append(row)
        self._write_policy_usage_csv()

    def _save_dynamic_policy_visuals(self, summary, epoch):
        if (
            not self.opt.is_main_process
            or not getattr(self.opt, "use_dynamic_policy", False)
            or summary is None
        ):
            return
        head_heatmap = summary.get("head_heatmap")
        if head_heatmap is not None:
            plt.figure(figsize=(10, 6))
            im = plt.imshow(
                head_heatmap,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
            )
            plt.xlabel("Head Index")
            plt.ylabel("Layer Index")
            plt.title(f"Average Head Keep Probability (best epoch={epoch})")
            cbar = plt.colorbar(im)
            cbar.set_label("average head keep probability")
            plt.tight_layout()
            plt.savefig(self.policy_heatmap_path, dpi=200)
            plt.close()

        block_curve = summary.get("per_layer_block_keep")
        if block_curve is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(list(range(len(block_curve))), block_curve, marker="o")
            plt.ylim(0.0, 1.05)
            plt.xlabel("Layer Index")
            plt.ylabel("Average block keep probability")
            plt.title(f"Average Block Keep Probability (best epoch={epoch})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.policy_block_curve_path, dpi=200)
            plt.close()

        token_example = summary.get("token_example")
        if token_example is not None:
            side = int(round(math.sqrt(token_example.shape[0])))
            if side * side == token_example.shape[0]:
                plt.figure(figsize=(5, 5))
                image = token_example.reshape(side, side)
                im = plt.imshow(image, cmap="viridis", interpolation="nearest")
                plt.title(f"Example Token Keep Probability (best epoch={epoch})")
                plt.axis("off")
                cbar = plt.colorbar(im)
                cbar.set_label("token keep probability")
                plt.tight_layout()
                plt.savefig(self.policy_token_example_path, dpi=200)
                plt.close()

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
        self.model.current_epoch = int(epoch)
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
        policy_meter = self._new_policy_meter()

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
            if getattr(self.opt, "use_dynamic_policy", False):
                policy_meter.update(self.model.get_dynamic_policy_state())
            last_lr = self.optimizer.param_groups[0]["lr"]

            if self.opt.is_main_process:
                desc = (
                    "Loss: %.3f, Focal: %.3f, Dice: %.3f, RFDiv: %.3f, LR: %.6f"
                    % (
                        _loss / (i + 1),
                        _focal_loss / (i + 1),
                        _dice_loss / (i + 1),
                        _rf_diversity / (i + 1),
                        last_lr,
                    )
                )
                if getattr(self.opt, "dino_lora_soft_gate", False):
                    soft_gate_aux = getattr(self.model, "last_aux_losses", {})
                    desc += (
                        ", SGRank: %.2f, SGRatio: %.3f, SGLambda: %.3f"
                        % (
                            float(
                                soft_gate_aux.get(
                                    "soft_gate_effective_rank_total", 0.0
                                )
                            ),
                            float(
                                soft_gate_aux.get(
                                    "soft_gate_expected_rank_ratio_mean", 0.0
                                )
                            ),
                            float(
                                soft_gate_aux.get(
                                    "soft_gate_budget_lambda", 0.0
                                )
                            ),
                        )
                    )
                if getattr(self.opt, "use_dynamic_policy", False):
                    dynamic_aux = getattr(self.model, "last_aux_losses", {})
                    desc += (
                        ", PolKeep: %.3f, PolCost: %.3f, PolLambda: %.3f"
                        % (
                            float(
                                dynamic_aux.get(
                                    "dynamic_policy_mean_head_keep", 1.0
                                )
                            ),
                            float(dynamic_aux.get("dynamic_policy_cost", 1.0)),
                            float(
                                dynamic_aux.get(
                                    "dynamic_policy_budget_lambda", 0.0
                                )
                            ),
                        )
                    )
                if getattr(self.opt, "cgla_temporal_reg_enable", False):
                    cgla_aux = getattr(self.model, "last_aux_losses", {})
                    desc += (
                        ", CGLAReg: %.3f, CGLALambda: %.3f"
                        % (
                            float(
                                cgla_aux.get("cgla_temporal_reg_loss", 0.0)
                            ),
                            float(
                                cgla_aux.get("cgla_temporal_reg_lambda", 0.0)
                            ),
                        )
                    )
                tbar.set_description(desc)

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
        soft_gate_aux = getattr(self.model, "last_aux_losses", {})
        self.last_train_policy_summary = policy_meter.summary()
        dynamic_policy_cost = (
            float(getattr(self.opt, "policy_head_weight", 1.0))
            * float(self.last_train_policy_summary.get("mean_head_keep", 1.0))
            + float(getattr(self.opt, "policy_block_weight", 0.0))
            * float(self.last_train_policy_summary.get("mean_block_keep", 1.0))
            + float(getattr(self.opt, "policy_token_weight", 0.0))
            * float(self.last_train_policy_summary.get("mean_token_keep", 1.0))
        )
        return {
            "loss": float(reduced[0].item() / denom),
            "focal": float(reduced[1].item() / denom),
            "dice": float(reduced[2].item() / denom),
            "rf_diversity": float(reduced[3].item() / denom),
            "lr": last_lr,
            "soft_gate_budget_lambda": float(
                soft_gate_aux.get("soft_gate_budget_lambda", 0.0)
            ),
            "soft_gate_budget_loss": float(
                soft_gate_aux.get("soft_gate_budget_loss", 0.0)
            ),
            "soft_gate_effective_rank_total": float(
                soft_gate_aux.get("soft_gate_effective_rank_total", 0.0)
            ),
            "soft_gate_expected_rank_ratio_mean": float(
                soft_gate_aux.get("soft_gate_expected_rank_ratio_mean", 0.0)
            ),
            "cgla_temporal_reg_lambda": float(
                soft_gate_aux.get("cgla_temporal_reg_lambda", 0.0)
            ),
            "cgla_temporal_reg_loss": float(
                soft_gate_aux.get("cgla_temporal_reg_loss", 0.0)
            ),
            "cgla_temporal_reg_layers": float(
                soft_gate_aux.get("cgla_temporal_reg_layers", 0.0)
            ),
            "cgla_temporal_reg_response_mean": float(
                soft_gate_aux.get("cgla_temporal_reg_response_mean", 0.0)
            ),
            "cgla_temporal_reg_change_loss": float(
                soft_gate_aux.get("cgla_temporal_reg_change_loss", 0.0)
            ),
            "cgla_temporal_reg_unchange_loss": float(
                soft_gate_aux.get("cgla_temporal_reg_unchange_loss", 0.0)
            ),
            "cgla_temporal_reg_mask_mean": float(
                soft_gate_aux.get("cgla_temporal_reg_mask_mean", 0.0)
            ),
            "cgla_temporal_reg_mask_nonzero_ratio": float(
                soft_gate_aux.get("cgla_temporal_reg_mask_nonzero_ratio", 0.0)
            ),
            "dynamic_policy_budget_lambda": float(
                soft_gate_aux.get("dynamic_policy_budget_lambda", 0.0)
            ),
            "dynamic_policy_budget_loss": float(
                soft_gate_aux.get("dynamic_policy_budget_loss", 0.0)
            ),
            "dynamic_policy_target_compute_ratio": float(
                soft_gate_aux.get("dynamic_policy_target_compute_ratio", 1.0)
            ),
            "dynamic_policy_cost": float(
                dynamic_policy_cost
            ),
            "dynamic_policy_mean_head_keep": float(
                self.last_train_policy_summary.get("mean_head_keep", 1.0)
                if self.last_train_policy_summary is not None
                else 1.0
            ),
            "dynamic_policy_head_active_ratio": float(
                self.last_train_policy_summary.get("head_active_ratio", 1.0)
                if self.last_train_policy_summary is not None
                else 1.0
            ),
            "dynamic_policy_mean_block_keep": float(
                self.last_train_policy_summary.get("mean_block_keep", 1.0)
                if self.last_train_policy_summary is not None
                else 1.0
            ),
            "dynamic_policy_mean_token_keep": float(
                self.last_train_policy_summary.get("mean_token_keep", 1.0)
                if self.last_train_policy_summary is not None
                else 1.0
            ),
        }

    def val(self, epoch):
        tbar = tqdm(self.val_data, ncols=80) if self.opt.is_main_process else self.val_data
        self.running_metric.clear()
        self.opt.phase = self.eval_phase
        self.model.eval()
        policy_meter = self._new_policy_meter()

        with torch.no_grad():
            for i, data in enumerate(tbar):
                val_pred = self.model.inference(
                    data["img1"].to(self.device, non_blocking=True),
                    data["img2"].to(self.device, non_blocking=True),
                )
                if getattr(self.opt, "use_dynamic_policy", False):
                    policy_meter.update(self.model.get_dynamic_policy_state())
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
                        self.eval_phase,
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
            if getattr(self.opt, "use_dynamic_policy", False):
                self.last_val_policy_summary = policy_meter.summary()
                policy_cost = (
                    float(getattr(self.opt, "policy_head_weight", 1.0))
                    * float(self.last_val_policy_summary.get("mean_head_keep", 1.0))
                    + float(getattr(self.opt, "policy_block_weight", 0.0))
                    * float(self.last_val_policy_summary.get("mean_block_keep", 1.0))
                    + float(getattr(self.opt, "policy_token_weight", 0.0))
                    * float(self.last_val_policy_summary.get("mean_token_keep", 1.0))
                )
                message += (
                    "mean_head_keep: %.3f head_active: %.3f policy_cost: %.3f "
                    % (
                        self.last_val_policy_summary.get("mean_head_keep", 1.0) * 100,
                        self.last_val_policy_summary.get("head_active_ratio", 1.0) * 100,
                        policy_cost * 100,
                    )
                )
            else:
                self.last_val_policy_summary = None
            print(message)
        else:
            self.last_val_policy_summary = policy_meter.summary() if getattr(
                self.opt, "use_dynamic_policy", False
            ) else None

        if self.opt.is_main_process:
            metric_value = self._safe_metric(val_scores, self.topk_metric_name)
            if math.isfinite(metric_value) and metric_value >= self.previous_best:
                self.previous_best = metric_value
                if getattr(self.opt, "use_dynamic_policy", False):
                    self._save_dynamic_policy_visuals(
                        self.last_val_policy_summary,
                        epoch,
                    )
            self._update_topk_checkpoints(epoch, val_scores)
            self._save_periodic_last(epoch, val_scores)
            self._write_checkpoint_index()

        return val_scores


if __name__ == "__main__":
    opt = Options().parse()
    init_distributed(opt)
    setup_seed(seed=1 + opt.rank)

    try:
        trainval = Trainval(opt)
        previous_train_loss = None
        spectral_search_update_count = 0

        for epoch in range(1, opt.num_epochs + 1):
            if opt.is_main_process:
                print(
                    "\n==> Name %s, Epoch %i, previous best = %.3f"
                    % (opt.name, epoch, trainval.previous_best * 100)
                )
            search_summary = trainval.model.update_lora_rank_search_with_val(
                epoch,
                trainval.lora_search_val_batches,
                trainval.lora_search_probe_batches,
            )
            if epoch == int(opt.num_epochs * 0.9):
                trainval._rescheduler(opt)
            train_stats = trainval.train(epoch)
            train_stats["spectral_first_search_jump"] = {}
            if (
                opt.is_main_process
                and getattr(opt, "dino_lora_search_spectral", False)
                and search_summary is not None
                and epoch > opt.dino_lora_search_warmup_epochs
            ):
                spectral_search_update_count += 1
                if spectral_search_update_count == 1:
                    previous_loss = (
                        float(previous_train_loss)
                        if previous_train_loss is not None
                        else float("nan")
                    )
                    loss_now = float(train_stats.get("loss", float("nan")))
                    if previous_train_loss is None:
                        loss_delta_pct = float("nan")
                    else:
                        denom = max(abs(float(previous_train_loss)), 1e-8)
                        loss_delta_pct = 100.0 * (loss_now - float(previous_train_loss)) / denom
                    train_stats["spectral_first_search_jump"] = {
                        "epoch": int(epoch),
                        "pre_active": int(
                            search_summary.get("pre_total_active_rank", -1)
                        ),
                        "post_active": int(
                            search_summary.get("total_active_rank", -1)
                        ),
                        "probe_blocks": int(
                            search_summary.get("probe_selected_blocks", -1)
                        ),
                        "probe_modules": int(
                            search_summary.get("probe_selected_modules", -1)
                        ),
                        "loss_prev": float(previous_loss),
                        "loss_now": float(loss_now),
                        "loss_delta_pct": float(loss_delta_pct),
                    }
                    print(
                        "Spectral first-search jump -> "
                        f"epoch={epoch}, "
                        f"pre_active={search_summary.get('pre_total_active_rank', -1)}, "
                        f"post_active={search_summary.get('total_active_rank', -1)}, "
                        f"probe_blocks={search_summary.get('probe_selected_blocks', -1)}, "
                        f"probe_modules={search_summary.get('probe_selected_modules', -1)}, "
                        f"loss_prev={previous_loss:.6f}, "
                        f"loss_now={loss_now:.6f}, "
                        f"loss_delta_pct={loss_delta_pct:.2f}%"
                    )
            val_scores = trainval.val(epoch)
            if getattr(opt, "use_dynamic_policy", False):
                trainval._record_policy_usage(
                    epoch,
                    "train",
                    trainval.last_train_policy_summary or {},
                    train_stats,
                )
                trainval._record_policy_usage(
                    epoch,
                    trainval.eval_phase,
                    trainval.last_val_policy_summary or {},
                    getattr(trainval.model, "last_aux_losses", {}),
                )
            trainval._append_log_line(epoch, train_stats, val_scores)
            previous_train_loss = float(train_stats.get("loss", float("nan")))
            if (
                opt.is_main_process
                and (
                    (
                        opt.dino_local_conv_rf_enable
                        and opt.dino_local_conv_rf_log_interval > 0
                        and epoch % opt.dino_local_conv_rf_log_interval == 0
                    )
                    or (
                        opt.mfce_rf_enable
                        and opt.mfce_rf_log_interval > 0
                        and epoch % opt.mfce_rf_log_interval == 0
                    )
                    or (
                        opt.decoder_rf_enable
                        and opt.decoder_rf_log_interval > 0
                        and epoch % opt.decoder_rf_log_interval == 0
                    )
                    or (
                        opt.pairlocal_rf_enable
                        and opt.pairlocal_rf_log_interval > 0
                        and epoch % opt.pairlocal_rf_log_interval == 0
                    )
                )
            ):
                trainval.model._log_rf_states(f"epoch {epoch}")

        if opt.is_main_process:
            print("Done!")
    finally:
        cleanup_distributed()
