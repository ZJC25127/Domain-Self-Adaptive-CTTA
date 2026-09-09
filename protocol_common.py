"""Shared implementation for the ImageNet+ and ImageNet++ entry points."""

import argparse
import json
import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from config import best_method_args, build_config
from model_loader import load_source_model
from moe_runtime import setup_moe


LOGGER = logging.getLogger("imagenet_protocol")
DATA_ROOT = os.environ.get("IMAGENET_DATA_ROOT")


def split_dataset_by_class(dataset, num_splits=3, seed=42):
    """Create the fixed per-class split used by ImageNet++."""
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for index, target in enumerate(dataset.targets):
        by_class[target].append(index)

    split_indices = [[] for _ in range(num_splits)]
    for indices in by_class.values():
        indices = np.asarray(indices)
        rng.shuffle(indices)
        split_size = len(indices) // num_splits
        for split_id in range(num_splits - 1):
            start = split_id * split_size
            split_indices[split_id].extend(indices[start:start + split_size])
        split_indices[-1].extend(indices[(num_splits - 1) * split_size:])
    return [Subset(dataset, indices) for indices in split_indices]


def _paths(data_root):
    return {
        "v2_top": os.path.join(data_root, "imagenet++/imagenetv2-top-images-format-val"),
        "v2_matched": os.path.join(
            data_root, "domain_gen/imagenetv2/imagenetv2-matched-frequency-format-val"
        ),
        "v2_threshold": os.path.join(
            data_root, "imagenet++/imagenetv2-threshold0.7-format-val"
        ),
        "a": os.path.join(data_root, "domain_gen/imagenet-adversarial/imagenet-a"),
        "s": os.path.join(data_root, "domain_gen/imagenet-sketch/images"),
        "r": os.path.join(data_root, "domain_gen/imagenet-rendition/imagenet-r"),
        "class_index": os.path.join(data_root, "imagenet_class_index.json"),
    }


def load_protocol_datasets(protocol, plus_v2_mode="full", data_root=None):
    data_root = data_root or DATA_ROOT
    if not data_root:
        raise ValueError("provide --data_root or set IMAGENET_DATA_ROOT")
    paths = _paths(data_root)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    v2_matched = datasets.ImageFolder(paths["v2_matched"], transform=transform)
    v2_top = datasets.ImageFolder(paths["v2_top"], transform=transform)
    v2_threshold = datasets.ImageFolder(paths["v2_threshold"], transform=transform)
    v2_plus_order = [v2_top, v2_matched, v2_threshold]
    v2_plusplus_order = [v2_matched, v2_top, v2_threshold]
    a = datasets.ImageFolder(paths["a"], transform=transform)
    s = datasets.ImageFolder(paths["s"], transform=transform)
    r = datasets.ImageFolder(paths["r"], transform=transform)

    with open(paths["class_index"], "r") as handle:
        index_to_info = json.load(handle)
    synset_to_index = {value[0]: int(key) for key, value in index_to_info.items()}

    def v2_indices(dataset):
        return [int(class_name) for class_name in dataset.classes]

    def synset_indices(dataset):
        return [synset_to_index[class_name] for class_name in dataset.classes]

    s_subsets = split_dataset_by_class(s)
    r_subsets = split_dataset_by_class(r)
    if protocol == "plus":
        phases = []
        if plus_v2_mode == "full":
            v2_full = ConcatDataset(v2_plus_order)
            for _ in range(3):
                phases.extend([
                    ("ImageNet-V2(full)", v2_full, v2_indices(v2_plus_order[0])),
                    ("ImageNet-A(full)", a, synset_indices(a)),
                    ("ImageNet-S(full)", s, synset_indices(s)),
                    ("ImageNet-R(full)", r, synset_indices(r)),
                ])
        elif plus_v2_mode == "separate":
            for round_id in range(3):
                v2_dataset = v2_plusplus_order[round_id]
                phases.extend([
                    (f"ImageNet-V2[{round_id}]", v2_dataset, v2_indices(v2_dataset)),
                    ("ImageNet-A(full)", a, synset_indices(a)),
                    ("ImageNet-S(full)", s, synset_indices(s)),
                    ("ImageNet-R(full)", r, synset_indices(r)),
                ])
        else:
            raise ValueError("plus_v2_mode must be 'full' or 'separate'")
        return phases

    if protocol == "plusplus":
        phases = []
        for round_id in range(3):
            phases.extend([
                (f"ImageNet-V2[{round_id}]", v2_plusplus_order[round_id],
                 v2_indices(v2_plusplus_order[round_id])),
                ("ImageNet-A(full)", a, synset_indices(a)),
                (f"ImageNet-S[{round_id}]", s_subsets[round_id], synset_indices(s)),
                (f"ImageNet-R[{round_id}]", r_subsets[round_id], synset_indices(r)),
            ])
        return phases
    raise ValueError("protocol must be 'plus' or 'plusplus'")


def run_phase(model, name, dataset, class_indices, batch_size, num_workers,
              device, max_batches=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    correct = 0
    total = 0
    for batch_id, (images, labels) in enumerate(tqdm(loader, desc=name, leave=False)):
        if max_batches is not None and batch_id >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs, _ = model(images, class_indices=class_indices)
        outputs = outputs.detach()
        for module in model.modules():
            if hasattr(module, "attention_out"):
                module.attention_out = None
        outputs = outputs[:, class_indices]
        correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    if total == 0:
        raise RuntimeError("phase %s produced no samples" % name)
    accuracy = correct / total
    error = 1.0 - accuracy
    LOGGER.info("phase=%s samples=%d accuracy=%.4f error=%.4f",
                name, total, accuracy, error)
    return {"name": name, "samples": total, "accuracy": accuracy, "error": error}


def _parse_overrides(items):
    values = {}
    for item in items:
        if "=" not in item:
            raise ValueError("override must have KEY=VALUE form: %s" % item)
        key, value = item.split("=", 1)
        values[key] = value
    return values


def run(protocol, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", choices=["best", "original"], default="best")
    parser.add_argument("--plus_v2_mode", choices=["full", "separate"], default="full")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--data_root", default=DATA_ROOT,
        help="root containing the ImageNet+, ImageNet++ directories",
    )
    parser.add_argument(
        "--checkpoint", default=os.environ.get("MOE_CHECKPOINT"),
        help="source ViT checkpoint (.pt); can also use MOE_CHECKPOINT",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()
    overrides = _parse_overrides(args.override)
    cfg = build_config(
        args.save_dir, args.config, overrides,
        data_root=args.data_root, checkpoint=args.checkpoint,
    )
    moe_args = best_method_args(args.config, overrides)
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "run.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet protocols require CUDA")
    device = torch.device("cuda")
    LOGGER.info("protocol=%s config=%s plus_v2_mode=%s device=%s",
                protocol, args.config, args.plus_v2_mode, device)
    LOGGER.info("batch_size=%d steps=%d", cfg.TEST.BATCH_SIZE, cfg.OPTIM.STEPS)

    base_model = load_source_model().cuda()
    model = setup_moe(moe_args, cfg, torch.nn.DataParallel(base_model).cuda())
    phases = load_protocol_datasets(
        protocol, args.plus_v2_mode, data_root=args.data_root,
    )
    LOGGER.info("phases=%d", len(phases))
    for name, dataset, class_indices in phases:
        LOGGER.info("dataset=%s samples=%d classes=%d", name, len(dataset), len(class_indices))

    results = []
    for phase_id, (name, dataset, class_indices) in enumerate(phases):
        round_id = phase_id // 4
        if round_id == 0:
            model.reset()
            LOGGER.info("reset before phase=%s", name)
        else:
            LOGGER.info("no reset before phase=%s", name)
        result = run_phase(
            model, name, dataset, class_indices, cfg.TEST.BATCH_SIZE,
            args.num_workers, device, args.max_batches,
        )
        result["round"] = round_id
        results.append(result)

    total = sum(item["samples"] for item in results)
    weighted_error = sum(item["error"] * item["samples"] for item in results) / total
    output = {
        "protocol": protocol,
        "config": args.config,
        "batch_size": cfg.TEST.BATCH_SIZE,
        "steps": cfg.OPTIM.STEPS,
        "overrides": overrides,
        "moe_args": vars(moe_args),
        "phase_results": results,
        "summary": {
            "samples": total,
            "error": weighted_error,
            "accuracy": 1.0 - weighted_error,
            "metric": "online_pre_update",
        },
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as handle:
        json.dump(output, handle, indent=2)
    LOGGER.info("FINAL online accuracy=%.4f error=%.4f",
                output["summary"]["accuracy"], output["summary"]["error"])
