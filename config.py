"""Configuration used by the two ImageNet online protocol entry points."""

import os
from types import SimpleNamespace

import yaml


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.join(ROOT, "cfgs", "vit", "moe_new.yaml")
DEFAULT_DATA_ROOT = os.environ.get("IMAGENET_DATA_ROOT")
DEFAULT_CHECKPOINT = os.environ.get("MOE_CHECKPOINT")


def _namespace(value):
    return SimpleNamespace(**value) if isinstance(value, dict) else value


def build_config(save_dir, config_name="best", overrides=None,
                 data_root=None, checkpoint=None):
    """Build a small config object matching the fields consumed by ``moe.py``."""
    overrides = overrides or {}
    with open(DEFAULT_YAML, "r") as handle:
        raw = yaml.safe_load(handle)

    data_root = data_root or DEFAULT_DATA_ROOT
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    if not data_root:
        raise ValueError("provide --data_root or set IMAGENET_DATA_ROOT")
    if not checkpoint:
        raise ValueError("provide --checkpoint or set MOE_CHECKPOINT")

    cfg = SimpleNamespace(
        MODEL=_namespace(raw.get("MODEL", {})),
        TEST=_namespace(raw.get("TEST", {})),
        OPTIM=_namespace(raw.get("OPTIM", {})),
        CORRUPTION=_namespace(raw.get("CORRUPTION", {})),
        DATA_DIR=data_root,
        SAVE_DIR=save_dir,
    )
    cfg.TEST.ckpt = checkpoint
    cfg.TEST.moe_new = True
    cfg.MODEL.ADAPTATION = "moe"
    cfg.CORRUPTION.DATASET = "imagenet"

    # PyYAML treats scientific notation without a decimal point as a string
    # in YAML 1.1; normalize the optimizer fields before constructing Adam.
    for field in ("LR", "MOELR", "BETA", "WD", "MT", "MT_MOE"):
        if hasattr(cfg.OPTIM, field):
            setattr(cfg.OPTIM, field, float(getattr(cfg.OPTIM, field)))
    for field in ("moe_rank", "moe_exp_num", "moe_router_num", "moe_top_k"):
        if hasattr(cfg.TEST, field):
            setattr(cfg.TEST, field, int(getattr(cfg.TEST, field)))
    cfg.TEST.adapter_scalar = float(cfg.TEST.adapter_scalar)

    if config_name in {"best", "imagenetc_best"}:
        cfg.TEST.adapter_scalar = 14.0
        cfg.OPTIM.BETA = 0.7
        cfg.OPTIM.MOELR = 3e-5
        if config_name == "imagenetc_best":
            cfg.TEST.adapter_scalar = 10.0
            cfg.OPTIM.BETA = 0.8
            cfg.OPTIM.MOELR = 1e-5
    elif config_name == "original":
        cfg.TEST.adapter_scalar = 10.0
        cfg.OPTIM.BETA = 0.9
        cfg.OPTIM.MOELR = 1e-5
    else:
        raise ValueError("config must be 'best' or 'original'")

    cfg.TEST.BATCH_SIZE = 200
    cfg.OPTIM.STEPS = 1
    for key, value in overrides.items():
        target = cfg.TEST if key in {
            "adapter_scalar", "moe_rank", "moe_exp_num", "moe_router_num",
            "moe_top_k",
        } else cfg.OPTIM if key in {"beta", "lr", "moelr"} else None
        field = {"beta": "BETA", "lr": "LR", "moelr": "MOELR"}.get(key, key)
        if target is None:
            continue
        if not hasattr(target, field):
            raise ValueError("unsupported config override: %s" % key)
        current = getattr(target, field)
        if isinstance(current, int) and not isinstance(current, bool):
            cast = int
        else:
            cast = float
        setattr(target, field, cast(value))
    return cfg


def best_method_args(config_name="best", overrides=None):
    """Return the MoE options used by the two online protocol entry points."""
    if config_name in {"best", "imagenetc_best"}:
        values = {
            "shared_ratio": 0.8,
            "adapter_dropout": 0.05,
            "disable_router_noise": True,
            "class_entropy_weight": 0.0,
            "sample_entropy_mode": "full",
            "sample_entropy_thresh": 2.6,
            "dynamic_entropy_thresh": False,
            "dynamic_entropy_quantile": 0.6,
            "dynamic_entropy_warmup": 5,
            "dynamic_entropy_cold_start": 0.9,
            "legacy_variance": False,
            "grad_clip_norm": 0.0,
            "restore_prob": 0.0,
            "sam_rho": 0.0,
            "redundancy_margin": 0.05,
            "eata_entropy_weighting": True,
            "eata_weight_scale": 1.25,
            "anchor_reg_weight": 0.2,
            "entropy_ratio": 1.0,
            "high_thresh": 2.5,
            "low_thresh": 1.5,
            "domain_feature_size": 96,
            "fix_ema_teacher": False,
            "deterministic_prediction": False,
            "post_update_prediction": False,
            "domain_slots": 14,
            "new_domain_init": "zero",
            "moe_layer_mode": "all",
        }
        if config_name == "imagenetc_best":
            values.update({
                "adapter_dropout": 0.1,
                "entropy_ratio": 0.9,
                "domain_feature_size": 72,
            })
    elif config_name == "original":
        values = {
            "shared_ratio": 0.9,
            "adapter_dropout": 0.1,
            "disable_router_noise": False,
            "class_entropy_weight": 0.0,
            "sample_entropy_mode": "full",
            "sample_entropy_thresh": None,
            "dynamic_entropy_thresh": False,
            "dynamic_entropy_quantile": 0.6,
            "dynamic_entropy_warmup": 5,
            "dynamic_entropy_cold_start": 0.9,
            "legacy_variance": False,
            "grad_clip_norm": 0.0,
            "restore_prob": 0.0,
            "sam_rho": 0.0,
            "redundancy_margin": 0.0,
            "eata_entropy_weighting": False,
            "eata_weight_scale": 1.0,
            "anchor_reg_weight": 0.0,
            "entropy_ratio": 0.6,
            "high_thresh": 2.5,
            "low_thresh": 1.5,
            "domain_feature_size": 72,
            "fix_ema_teacher": False,
            "deterministic_prediction": False,
            "post_update_prediction": False,
            "domain_slots": 14,
            "new_domain_init": "zero",
            "moe_layer_mode": "all",
        }
    else:
        raise ValueError("config must be 'best' or 'original'")

    for key, value in (overrides or {}).items():
        if key not in values:
            continue
        old = values[key]
        if isinstance(old, bool):
            values[key] = str(value).lower() in {"1", "true", "yes", "on"}
        elif isinstance(old, int) and not isinstance(old, bool):
            values[key] = int(value)
        elif isinstance(old, float):
            values[key] = float(value)
        elif old is None:
            values[key] = None if str(value).lower() == "none" else float(value)
        else:
            values[key] = value
    return SimpleNamespace(**values)
