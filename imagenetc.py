"""Run the ImageNet-C continual test-time adaptation protocol."""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config import best_method_args, build_config
from model_loader import load_source_model
from moe_runtime import setup_moe


LOGGER = logging.getLogger("imagenetc")

CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)


class ImageNetCDataset(Dataset):
    """ImageNet-C samples in the original RobustBench evaluation order."""

    def __init__(self, data_root, corruption, severity, num_examples,
                 metadata_root):
        self.root = os.path.join(data_root, "ImageNet-C", corruption, str(severity))
        ids_path = os.path.join(metadata_root, "imagenet_test_image_ids.txt")
        map_path = os.path.join(metadata_root, "imagenet_class_to_id_map.json")
        with open(ids_path, "r") as handle:
            image_ids = [line.strip() for line in handle if line.strip()]
        with open(map_path, "r") as handle:
            class_to_id = json.load(handle)
        self.samples = [(os.path.join(self.root, image_id), class_to_id[image_id.split("/")[0]])
                        for image_id in image_ids[:num_examples]]
        missing = [path for path, _ in self.samples if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "ImageNet-C is incomplete; missing example: %s" % missing[0]
            )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, "rb") as handle:
            image = Image.open(handle).convert("RGB")
        view = self.transform(image)
        return view, target


def evaluate_corruption(model, name, dataset, batch_size, num_workers,
                        device, class_indices, max_batches=None, only_test=False):
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
    label = name + (" [post-update]" if only_test else "")
    for batch_id, (images, labels) in enumerate(tqdm(loader, desc=label, leave=False)):
        if max_batches is not None and batch_id >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs, _ = model(images, only_test=only_test, class_indices=class_indices)
        outputs = outputs.detach()
        for module in model.modules():
            if hasattr(module, "attention_out"):
                module.attention_out = None
        outputs = outputs[:, class_indices]
        correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    if total == 0:
        raise RuntimeError("corruption %s produced no samples" % name)
    accuracy = correct / total
    result = {
        "corruption": name,
        "samples": total,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
    }
    LOGGER.info("corruption=%s samples=%d accuracy=%.4f error=%.4f",
                name, total, result["accuracy"], result["error"])
    return result


def parse_overrides(items):
    values = {}
    for item in items:
        if "=" not in item:
            raise ValueError("override must have KEY=VALUE form: %s" % item)
        key, value = item.split("=", 1)
        values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["original", "best"], default="best")
    parser.add_argument("--data_root", default=os.environ.get("IMAGENET_DATA_ROOT"),
                        help="parent directory containing ImageNet-C/")
    parser.add_argument("--checkpoint", default=os.environ.get("MOE_CHECKPOINT"),
                        help="source ViT checkpoint (.pt)")
    parser.add_argument("--metadata_root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "metadata"))
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--num_examples", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="optional smoke-test limit per corruption")
    parser.add_argument("--post_corruption_eval", action="store_true",
                        help="also evaluate each corruption after its updates")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    config_name = "imagenetc_best" if args.config == "best" else "original"
    cfg = build_config(
        args.save_dir, config_name, overrides,
        data_root=args.data_root, checkpoint=args.checkpoint,
    )
    cfg.TEST.BATCH_SIZE = 50
    cfg.OPTIM.STEPS = 1
    moe_args = best_method_args(config_name, overrides)
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
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet-C requires CUDA")
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    device = torch.device("cuda")
    LOGGER.info("protocol=imagenetc config=%s device=%s", args.config, device)
    LOGGER.info("severity=%d examples=%d batch_size=%d steps=%d",
                args.severity, args.num_examples, cfg.TEST.BATCH_SIZE, cfg.OPTIM.STEPS)
    LOGGER.info("moe_args=%s", vars(moe_args))

    base_model = load_source_model().cuda()
    model = setup_moe(moe_args, cfg, torch.nn.DataParallel(base_model).cuda())
    class_indices = list(range(1000))
    results = []
    post_results = []
    for corruption_id, corruption in enumerate(CORRUPTIONS):
        dataset = ImageNetCDataset(
            args.data_root, corruption, args.severity, args.num_examples,
            args.metadata_root,
        )
        if corruption_id == 0:
            model.reset()
            LOGGER.info("reset before corruption=%s", corruption)
        else:
            LOGGER.info("no reset before corruption=%s", corruption)
        result = evaluate_corruption(
            model, "%s%d" % (corruption, args.severity), dataset,
            cfg.TEST.BATCH_SIZE, args.num_workers, device, class_indices,
            args.max_batches,
        )
        results.append(result)
        if args.post_corruption_eval:
            post_results.append(evaluate_corruption(
                model, "%s%d" % (corruption, args.severity), dataset,
                cfg.TEST.BATCH_SIZE, args.num_workers, device, class_indices,
                args.max_batches, only_test=True,
            ))

    def summarize(items, metric):
        total = sum(item["samples"] for item in items)
        error = sum(item["error"] * item["samples"] for item in items) / total
        return {"samples": total, "accuracy": 1.0 - error,
                "error": error, "metric": metric}

    output = {
        "protocol": "imagenetc",
        "config": args.config,
        "severity": args.severity,
        "num_examples": args.num_examples,
        "batch_size": cfg.TEST.BATCH_SIZE,
        "steps": cfg.OPTIM.STEPS,
        "overrides": overrides,
        "moe_args": vars(moe_args),
        "corruptions": list(CORRUPTIONS),
        "results": results,
        "summary": summarize(results, "online_pre_update"),
    }
    if post_results:
        output["post_results"] = post_results
        output["post_summary"] = summarize(post_results, "post_update_only_test")
    with open(os.path.join(args.save_dir, "results.json"), "w") as handle:
        json.dump(output, handle, indent=2)
    LOGGER.info("FINAL online accuracy=%.4f error=%.4f",
                output["summary"]["accuracy"], output["summary"]["error"])
    if post_results:
        LOGGER.info("FINAL post-update accuracy=%.4f error=%.4f",
                    output["post_summary"]["accuracy"], output["post_summary"]["error"])


if __name__ == "__main__":
    main()
