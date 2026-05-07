import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.cd_dataset import DataLoader
from model.blocks.adapter import DINOV3Wrapper
from model.create_ChangeDINO import Model as ChangeDINOModelWrapper
from option import Options
from util.metric_tool import ConfuseMatrixMeter, cm2score


CSV_FIELDNAMES = [
    "layer_index",
    "head_index",
    "num_heads",
    "head_dim",
    "baseline_iou_1",
    "masked_iou_1",
    "drop_iou_1",
    "baseline_F1_1",
    "masked_F1_1",
    "drop_F1_1",
    "baseline_miou",
    "masked_miou",
    "drop_miou",
    "baseline_mf1",
    "masked_mf1",
    "drop_mf1",
    "baseline_acc",
    "masked_acc",
    "drop_acc",
]
SUMMARY_METRIC_KEYS = [
    "acc",
    "miou",
    "mf1",
    "iou_0",
    "iou_1",
    "F1_0",
    "F1_1",
    "precision_0",
    "precision_1",
    "recall_0",
    "recall_1",
]
PRIMARY_DROP_KEYS = ("iou_1", "F1_1", "miou")


def build_head_ablation_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--head_ablation_enable", action="store_true")
    parser.add_argument("--head_ablation_max_batches", type=int, default=10)
    parser.add_argument("--head_ablation_layers", type=str, default="all")
    parser.add_argument("--head_ablation_heads", type=str, default="all")
    parser.add_argument("--head_ablation_metric", type=str, default="iou_1")
    parser.add_argument("--head_ablation_output_dir", type=str, default="")
    parser.add_argument("--head_ablation_smoke", action="store_true")
    parser.add_argument("--head_ablation_ckpt", type=str, default="")
    return parser


def parse_options_and_ablation_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ablation_parser = build_head_ablation_parser()
    ablation_args, remaining_argv = ablation_parser.parse_known_args(argv)

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining_argv
        opt = Options().parse()
    finally:
        sys.argv = original_argv

    return opt, ablation_args


def configure_single_process_eval(opt):
    opt.phase = "val"
    opt.distributed = False
    opt.world_size = 1
    opt.rank = 0
    opt.local_rank = 0
    opt.is_main_process = True
    if getattr(opt, "gpu_ids", None):
        opt.gpu_ids = [opt.gpu_ids[0]]
        opt.device_id = opt.gpu_ids[0]
        if torch.cuda.is_available():
            torch.cuda.set_device(opt.device_id)
    else:
        opt.device_id = -1
    return opt


def unwrap_parallel(module):
    while isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        module = module.module
    return module


def iter_named_modules_dedup(root):
    stack = [("", root)]
    visited = set()
    while stack:
        prefix, module = stack.pop()
        module = unwrap_parallel(module)
        module_id = id(module)
        if module_id in visited:
            continue
        visited.add(module_id)
        yield prefix, module
        children = list(module.named_children())
        for child_name, child in reversed(children):
            next_prefix = f"{prefix}.{child_name}" if prefix else child_name
            stack.append((next_prefix, child))


def preview_module_tree(root, max_depth=4, max_lines=40):
    lines = []
    for name, module in iter_named_modules_dedup(root):
        depth = 0 if not name else name.count(".") + 1
        if depth > max_depth:
            continue
        label = name or "<root>"
        lines.append(f"{'  ' * depth}{label}: {module.__class__.__name__}")
        if len(lines) >= max_lines:
            lines.append("  ...")
            break
    return "\n".join(lines)


def find_dino_wrapper(model_wrapper):
    for name, module in iter_named_modules_dedup(model_wrapper):
        if isinstance(module, DINOV3Wrapper):
            return module, (name or "<root>")
    raise RuntimeError(
        "Failed to locate DINOV3Wrapper inside the current model.\n"
        "Model tree preview:\n"
        f"{preview_module_tree(model_wrapper)}"
    )


def get_underlying_dino_block(block):
    while hasattr(block, "block"):
        block = block.block
    return block


def resolve_attention_module(block, layer_index):
    attn = getattr(block, "attn", None)
    if attn is not None:
        return attn
    attn = getattr(block, "attention", None)
    if attn is not None:
        return attn
    raise RuntimeError(
        f"DINO block at layer {layer_index} does not expose .attn or .attention. "
        f"Block type: {type(block).__name__}"
    )


def infer_num_heads(attn_module, layer_index):
    for attr_name in ("num_heads", "heads", "n_heads"):
        value = getattr(attn_module, attr_name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Failed to convert {attr_name}={value!r} to int for attention module "
                f"at layer {layer_index} ({type(attn_module).__name__})."
            )
    head_like_attrs = sorted(name for name in dir(attn_module) if "head" in name.lower())
    raise RuntimeError(
        f"Could not infer num_heads for attention module at layer {layer_index} "
        f"({type(attn_module).__name__}). Head-like attributes: {head_like_attrs}"
    )


def collect_dino_attention_heads(model_wrapper):
    dino_wrapper, dino_wrapper_name = find_dino_wrapper(model_wrapper)
    blocks = getattr(dino_wrapper.model, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            f"Located DINOV3Wrapper at {dino_wrapper_name}, but it has no .model.blocks."
        )

    module_name_map = {id(module): (name or "<root>") for name, module in dino_wrapper.named_modules()}
    head_specs = []
    per_layer_num_heads = {}
    for layer_index, block in enumerate(blocks):
        base_block = get_underlying_dino_block(block)
        attn_module = resolve_attention_module(base_block, layer_index)
        proj_module = getattr(attn_module, "proj", None)
        if proj_module is None:
            raise RuntimeError(
                f"Attention module at layer {layer_index} ({type(attn_module).__name__}) "
                "does not expose .proj."
            )

        num_heads = infer_num_heads(attn_module, layer_index)
        if not hasattr(proj_module, "in_features"):
            raise RuntimeError(
                f"Attention proj module at layer {layer_index} ({type(proj_module).__name__}) "
                "does not expose in_features."
            )
        if proj_module.in_features % num_heads != 0:
            raise RuntimeError(
                f"proj.in_features={proj_module.in_features} is not divisible by num_heads={num_heads} "
                f"for layer {layer_index}."
            )
        head_dim = int(proj_module.in_features // num_heads)
        per_layer_num_heads[layer_index] = num_heads
        attn_module_name = module_name_map.get(id(attn_module), f"model.blocks.{layer_index}.attn")

        for head_index in range(num_heads):
            head_specs.append(
                {
                    "layer_index": int(layer_index),
                    "head_index": int(head_index),
                    "num_heads": int(num_heads),
                    "head_dim": int(head_dim),
                    "proj_module": proj_module,
                    "attn_module_name": attn_module_name,
                }
            )

    total_layers = len(per_layer_num_heads)
    total_heads = len(head_specs)
    expected_total = 24 * 16
    print(
        f"Located DINOV3Wrapper at {dino_wrapper_name} with {total_layers} blocks and "
        f"{total_heads} attention heads."
    )
    if total_heads != expected_total:
        print(
            f"Head count differs from the expected ViT-L/16 layout ({expected_total}); "
            f"using the actual discovered count."
        )
    return head_specs


def parse_index_spec(spec, label):
    spec = str(spec).strip().lower()
    if spec == "all":
        return None
    indices = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid {label} range '{chunk}'.")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(chunk))
    if not indices:
        raise ValueError(f"{label} selection is empty: {spec!r}")
    if any(index < 0 for index in indices):
        raise ValueError(f"{label} indices must be non-negative: {spec!r}")
    return sorted(indices)


def select_head_specs(head_specs, layer_spec, head_spec, smoke=False):
    selected_layers = parse_index_spec(layer_spec, "layer")
    selected_heads = parse_index_spec(head_spec, "head")

    selected = []
    for head_info in head_specs:
        if selected_layers is not None and head_info["layer_index"] not in selected_layers:
            continue
        if selected_heads is not None and head_info["head_index"] not in selected_heads:
            continue
        selected.append(head_info)

    if smoke:
        smoke_layers = sorted({item["layer_index"] for item in selected})[:2]
        smoke_selected = []
        for layer_index in smoke_layers:
            layer_heads = [item for item in selected if item["layer_index"] == layer_index]
            keep_heads = sorted({item["head_index"] for item in layer_heads})[:2]
            smoke_selected.extend(
                item for item in layer_heads if item["head_index"] in keep_heads
            )
        selected = smoke_selected

    selected.sort(key=lambda item: (item["layer_index"], item["head_index"]))
    return selected


def metrics_to_float_dict(metrics):
    return {key: float(metrics[key]) for key in SUMMARY_METRIC_KEYS if key in metrics}


def format_metrics(metrics):
    ordered = []
    for key in SUMMARY_METRIC_KEYS:
        if key in metrics:
            ordered.append(f"{key}={float(metrics[key]):.4f}")
    return ", ".join(ordered)


def evaluate_model(model_wrapper, dataloader, device, max_batches=None):
    running_metric = ConfuseMatrixMeter(n_class=2)
    model_wrapper.eval()

    with torch.no_grad():
        for batch_index, data in enumerate(dataloader):
            if max_batches is not None and max_batches > 0 and batch_index >= max_batches:
                break
            pred = model_wrapper.inference(
                data["img1"].to(device, non_blocking=True),
                data["img2"].to(device, non_blocking=True),
            )
            pred = torch.argmax(pred.detach(), dim=1)
            target = data["cd_label"].detach()
            running_metric.update_cm(
                pr=pred.cpu().numpy(),
                gt=target.cpu().numpy(),
            )

    confusion = (
        running_metric.sum
        if running_metric.initialized
        else np.zeros((2, 2), dtype=np.float64)
    )
    return metrics_to_float_dict(cm2score(confusion))


def build_head_mask_hook(start, end, debug_prefix=""):
    def hook(module, inputs):
        if not inputs:
            raise RuntimeError(f"{debug_prefix} attention proj forward_pre_hook received empty inputs.")
        x = inputs[0]
        y = x.clone()
        y[..., start:end] = 0
        return (y,) + inputs[1:]

    return hook


def summarize_checkpoint_keys(keys, limit=5):
    keys = list(keys)
    if not keys:
        return "[]"
    preview = keys[:limit]
    suffix = "" if len(keys) <= limit else f" ... (+{len(keys) - limit} more)"
    return f"{preview}{suffix}"


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "network", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
            first_value = next(iter(checkpoint.values()))
            if torch.is_tensor(first_value):
                return checkpoint
    raise RuntimeError(
        "Unsupported checkpoint format. Expected a dict with one of "
        "{'model_state_dict', 'network', 'state_dict'} or a raw state_dict."
    )


def maybe_strip_module_prefix(state_dict, target_keys):
    if not state_dict:
        return state_dict
    has_module_prefix = all(key.startswith("module.") for key in state_dict.keys())
    target_has_module_prefix = any(key.startswith("module.") for key in target_keys)
    if has_module_prefix and not target_has_module_prefix:
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_into_model(model_wrapper, checkpoint_path):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    network = model_wrapper._unwrap_model(model_wrapper.model)
    state_dict = maybe_strip_module_prefix(state_dict, network.state_dict().keys())
    load_result = network.load_state_dict(state_dict, strict=False)

    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(
        f"missing_keys={len(missing_keys)} {summarize_checkpoint_keys(missing_keys)} | "
        f"unexpected_keys={len(unexpected_keys)} {summarize_checkpoint_keys(unexpected_keys)}"
    )
    return checkpoint_path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_existing_csv_records(csv_path):
    if not os.path.isfile(csv_path):
        return {}

    records = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            parsed = {}
            for key, value in row.items():
                if key in {"layer_index", "head_index", "num_heads", "head_dim"}:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            records[(parsed["layer_index"], parsed["head_index"])] = parsed
    return records


def make_csv_row(head_info, baseline_metrics, masked_metrics):
    row = {
        "layer_index": int(head_info["layer_index"]),
        "head_index": int(head_info["head_index"]),
        "num_heads": int(head_info["num_heads"]),
        "head_dim": int(head_info["head_dim"]),
    }
    for metric_name in ("iou_1", "F1_1", "miou", "mf1", "acc"):
        baseline_value = float(baseline_metrics[metric_name])
        masked_value = float(masked_metrics[metric_name])
        row[f"baseline_{metric_name}"] = baseline_value
        row[f"masked_{metric_name}"] = masked_value
        row[f"drop_{metric_name}"] = baseline_value - masked_value
    return row


def make_result_record(csv_row, head_info, masked_metrics, baseline_metrics):
    drop_metrics = {
        metric_name: float(baseline_metrics[metric_name] - masked_metrics[metric_name])
        for metric_name in baseline_metrics.keys()
        if metric_name in masked_metrics
    }
    return {
        **csv_row,
        "attn_module_name": head_info["attn_module_name"],
        "masked_metrics": metrics_to_float_dict(masked_metrics),
        "baseline_metrics": metrics_to_float_dict(baseline_metrics),
        "drop_metrics": metrics_to_float_dict(drop_metrics),
    }


def save_ranked_csv(records, path, sort_key, reverse=False):
    ordered = sorted(
        records,
        key=lambda item: (
            float(item.get(sort_key, float("nan"))),
            int(item["layer_index"]),
            int(item["head_index"]),
        ),
        reverse=reverse,
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row[field] for field in CSV_FIELDNAMES})


def build_heatmap(records, num_layers, max_num_heads, drop_key):
    heatmap = np.full((num_layers, max_num_heads), np.nan, dtype=np.float32)
    for record in records:
        layer_index = int(record["layer_index"])
        head_index = int(record["head_index"])
        if 0 <= layer_index < num_layers and 0 <= head_index < max_num_heads:
            heatmap[layer_index, head_index] = float(record[drop_key])
    return heatmap


def plot_heatmap(heatmap, metric_name, baseline_value, max_batches, output_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#d9d9d9")
    image = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap=cmap, origin="upper")
    ax.set_xlabel("head index")
    ax.set_ylabel("layer index")
    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_yticks(np.arange(heatmap.shape[0]))
    title = (
        f"DINO head ablation {metric_name} drop | baseline {metric_name}={baseline_value:.4f} | "
        f"max_batches={max_batches} | masking-based analysis only"
    )
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("metric drop after masking one head")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def ensure_csv_writer(csv_path):
    file_exists = os.path.isfile(csv_path)
    handle = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
    if not file_exists or os.path.getsize(csv_path) == 0:
        writer.writeheader()
        handle.flush()
    return handle, writer


def validate_resume_compatibility(csv_path, baseline_path, requested_max_batches):
    if not os.path.isfile(csv_path):
        return
    if not os.path.isfile(baseline_path):
        return
    payload = load_json(baseline_path)
    saved_max_batches = int(payload.get("max_batches", requested_max_batches))
    if saved_max_batches != requested_max_batches:
        raise RuntimeError(
            "Existing head ablation outputs were created with a different "
            f"--head_ablation_max_batches ({saved_max_batches} vs {requested_max_batches}). "
            "Use a different output directory or remove the old outputs."
        )


def print_smoke_head_details(selected_heads):
    print("Smoke mode head selection:")
    for head_info in selected_heads:
        start = head_info["head_index"] * head_info["head_dim"]
        end = start + head_info["head_dim"]
        print(
            f"  layer={head_info['layer_index']} head={head_info['head_index']} "
            f"num_heads={head_info['num_heads']} head_dim={head_info['head_dim']} "
            f"slice=[{start}:{end}] attn={head_info['attn_module_name']}"
        )


def main():
    opt, args = parse_options_and_ablation_args()
    explicit_ckpt = os.path.abspath(args.head_ablation_ckpt) if args.head_ablation_ckpt else ""
    if explicit_ckpt:
        opt.load_pretrain = False
    configure_single_process_eval(opt)

    model_wrapper = ChangeDINOModelWrapper(opt)
    if explicit_ckpt:
        loaded_checkpoint_path = load_checkpoint_into_model(model_wrapper, explicit_ckpt)
    elif opt.load_pretrain:
        default_ckpt = os.path.join(model_wrapper.save_dir, f"{opt.name}_{opt.backbone}_best.pth")
        loaded_checkpoint_path = os.path.abspath(default_ckpt)
    else:
        loaded_checkpoint_path = ""
        print(
            "No checkpoint was explicitly loaded. Running analysis on the current in-memory model weights."
        )

    device = model_wrapper.device
    val_loader = DataLoader(opt).load_data()
    print(f"#validation batches = {len(val_loader)}")
    print("Analysis note: masking-based analysis only. Attention computation still happens.")

    all_head_specs = collect_dino_attention_heads(model_wrapper)
    selected_heads = select_head_specs(
        all_head_specs,
        layer_spec=args.head_ablation_layers,
        head_spec=args.head_ablation_heads,
        smoke=args.head_ablation_smoke,
    )
    if not selected_heads:
        raise RuntimeError("No DINO attention heads matched the requested selection.")
    print(f"Selected {len(selected_heads)} heads for ablation.")
    if args.head_ablation_smoke:
        print_smoke_head_details(selected_heads)

    output_dir = (
        os.path.abspath(args.head_ablation_output_dir)
        if args.head_ablation_output_dir
        else os.path.join(model_wrapper.save_dir, "head_ablation")
    )
    os.makedirs(output_dir, exist_ok=True)

    baseline_path = os.path.join(output_dir, "baseline_metrics.json")
    results_csv_path = os.path.join(output_dir, "head_ablation_results.csv")
    results_json_path = os.path.join(output_dir, "head_ablation_results.json")
    validate_resume_compatibility(
        results_csv_path,
        baseline_path,
        requested_max_batches=args.head_ablation_max_batches,
    )

    baseline_payload = None
    if os.path.isfile(baseline_path):
        baseline_payload = load_json(baseline_path)
        saved_max_batches = int(baseline_payload.get("max_batches", args.head_ablation_max_batches))
        if saved_max_batches != args.head_ablation_max_batches:
            baseline_payload = None

    if baseline_payload is None:
        baseline_metrics = evaluate_model(
            model_wrapper=model_wrapper,
            dataloader=val_loader,
            device=device,
            max_batches=args.head_ablation_max_batches,
        )
        baseline_payload = {
            **baseline_metrics,
            "max_batches": int(args.head_ablation_max_batches),
            "phase": "val",
            "checkpoint_path": loaded_checkpoint_path,
            "analysis_mode": "masking-based analysis only",
        }
        save_json(baseline_path, baseline_payload)
    else:
        if "metrics" in baseline_payload and isinstance(baseline_payload["metrics"], dict):
            baseline_metrics = metrics_to_float_dict(baseline_payload["metrics"])
        else:
            baseline_metrics = metrics_to_float_dict(baseline_payload)

    print(f"Baseline metrics: {format_metrics(baseline_metrics)}")
    if args.head_ablation_metric not in baseline_metrics:
        raise ValueError(
            f"--head_ablation_metric={args.head_ablation_metric!r} is not available. "
            f"Choose from: {sorted(baseline_metrics.keys())}"
        )

    existing_csv_records = load_existing_csv_records(results_csv_path)
    completed_pairs = set(existing_csv_records.keys())
    pending_heads = [
        head_info
        for head_info in selected_heads
        if (head_info["layer_index"], head_info["head_index"]) not in completed_pairs
    ]
    print(
        f"Resume state: {len(completed_pairs)} completed, {len(pending_heads)} pending "
        f"within the current selection."
    )

    csv_handle, csv_writer = ensure_csv_writer(results_csv_path)
    try:
        progress = tqdm(pending_heads, ncols=120)
        for head_info in progress:
            layer_index = head_info["layer_index"]
            head_index = head_info["head_index"]
            head_dim = head_info["head_dim"]
            start = head_index * head_dim
            end = start + head_dim
            debug_prefix = f"[layer={layer_index} head={head_index}]"
            hook = head_info["proj_module"].register_forward_pre_hook(
                build_head_mask_hook(start, end, debug_prefix=debug_prefix)
            )
            start_time = time.time()
            try:
                masked_metrics = evaluate_model(
                    model_wrapper=model_wrapper,
                    dataloader=val_loader,
                    device=device,
                    max_batches=args.head_ablation_max_batches,
                )
            finally:
                hook.remove()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            csv_row = make_csv_row(head_info, baseline_metrics, masked_metrics)
            csv_writer.writerow(csv_row)
            csv_handle.flush()

            existing_csv_records[(layer_index, head_index)] = csv_row
            elapsed = time.time() - start_time
            primary_metric = args.head_ablation_metric
            masked_value = float(masked_metrics.get(primary_metric, float("nan")))
            drop_value = float(csv_row.get(f"drop_{primary_metric}", float("nan")))
            progress.set_description(
                f"layer {layer_index:02d} head {head_index:02d}"
            )
            progress.set_postfix(
                masked_metric=f"{masked_value:.4f}",
                drop=f"{drop_value:.4f}",
                elapsed=f"{elapsed:.1f}s",
            )
    finally:
        csv_handle.close()

    final_rows = [
        existing_csv_records[key]
        for key in sorted(existing_csv_records.keys(), key=lambda item: (item[0], item[1]))
    ]
    selected_pairs = {
        (head_info["layer_index"], head_info["head_index"]): head_info for head_info in selected_heads
    }
    full_results = []
    for row in final_rows:
        pair = (row["layer_index"], row["head_index"])
        head_info = selected_pairs.get(pair)
        if head_info is None:
            head_info = next(
                info
                for info in all_head_specs
                if info["layer_index"] == row["layer_index"] and info["head_index"] == row["head_index"]
            )
        masked_metrics = {
            "iou_1": row["masked_iou_1"],
            "F1_1": row["masked_F1_1"],
            "miou": row["masked_miou"],
            "mf1": row["masked_mf1"],
            "acc": row["masked_acc"],
        }
        full_results.append(
            make_result_record(
                csv_row=row,
                head_info=head_info,
                masked_metrics=masked_metrics,
                baseline_metrics=baseline_metrics,
            )
        )

    results_payload = {
        "baseline": baseline_payload,
        "checkpoint_path": loaded_checkpoint_path,
        "phase": "val",
        "max_batches": int(args.head_ablation_max_batches),
        "analysis_mode": "masking-based analysis only",
        "selected_layers": args.head_ablation_layers,
        "selected_heads": args.head_ablation_heads,
        "smoke": bool(args.head_ablation_smoke),
        "results": full_results,
    }
    save_json(results_json_path, results_payload)

    num_layers = max(item["layer_index"] for item in all_head_specs) + 1
    max_num_heads = max(item["num_heads"] for item in all_head_specs)
    heatmap_specs = [
        ("iou_1", "drop_iou_1", "heatmap_iou_1_drop.npy", "heatmap_iou_1_drop.png"),
        ("F1_1", "drop_F1_1", "heatmap_F1_1_drop.npy", "heatmap_F1_1_drop.png"),
        ("miou", "drop_miou", "heatmap_miou_drop.npy", "heatmap_miou_drop.png"),
    ]
    for metric_name, drop_key, npy_name, png_name in heatmap_specs:
        heatmap = build_heatmap(full_results, num_layers, max_num_heads, drop_key)
        np.save(os.path.join(output_dir, npy_name), heatmap)
        plot_heatmap(
            heatmap=heatmap,
            metric_name=metric_name,
            baseline_value=float(baseline_metrics[metric_name]),
            max_batches=args.head_ablation_max_batches,
            output_path=os.path.join(output_dir, png_name),
        )

    save_ranked_csv(
        final_rows,
        os.path.join(output_dir, "redundant_heads_iou_1.csv"),
        sort_key="drop_iou_1",
        reverse=False,
    )
    save_ranked_csv(
        final_rows,
        os.path.join(output_dir, "important_heads_iou_1.csv"),
        sort_key="drop_iou_1",
        reverse=True,
    )

    print(f"Saved outputs to: {output_dir}")
    print(
        f"Most important head by iou_1 drop: "
        f"{max(final_rows, key=lambda item: item['drop_iou_1'])['layer_index']}/"
        f"{max(final_rows, key=lambda item: item['drop_iou_1'])['head_index']}"
    )
    print(
        f"Least important head by iou_1 drop: "
        f"{min(final_rows, key=lambda item: item['drop_iou_1'])['layer_index']}/"
        f"{min(final_rows, key=lambda item: item['drop_iou_1'])['head_index']}"
    )


if __name__ == "__main__":
    main()
