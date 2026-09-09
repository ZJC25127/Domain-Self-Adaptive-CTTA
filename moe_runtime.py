"""MoE construction and optimizer wiring shared by ImageNet+ and ++."""

import logging

import torch
import torch.optim as optim

import moe


LOGGER = logging.getLogger(__name__)


def setup_moe(args, cfg, model):
    model = moe.configure_model(
        model,
        cfg,
        shared_ratio=args.shared_ratio,
        adapter_dropout=args.adapter_dropout,
        noisy_gating=not args.disable_router_noise,
        domain_slots=args.domain_slots,
        moe_layer_mode=args.moe_layer_mode,
    )
    model_params, moe_params = moe.collect_params(model, freeze=True)
    optimizer = setup_optimizer_moe(
        model_params, moe_params, cfg.OPTIM.LR, cfg.OPTIM.MOELR, cfg.OPTIM
    )
    wrapped = moe.MOE(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=False,
        ema=cfg.OPTIM.MT,
        ema_moe=cfg.OPTIM.MT_MOE,
        legacy_variance=args.legacy_variance,
        class_entropy_weight=args.class_entropy_weight,
        sample_entropy_mode=args.sample_entropy_mode,
        sample_entropy_thresh=args.sample_entropy_thresh,
        dynamic_entropy_thresh=args.dynamic_entropy_thresh,
        dynamic_entropy_quantile=args.dynamic_entropy_quantile,
        dynamic_entropy_warmup=args.dynamic_entropy_warmup,
        dynamic_entropy_cold_start=args.dynamic_entropy_cold_start,
        aug_loss_weight=args.aug_loss_weight,
        aug_objective=args.aug_objective,
        grad_clip_norm=args.grad_clip_norm,
        restore_prob=args.restore_prob,
        sam_rho=args.sam_rho,
        redundancy_margin=args.redundancy_margin,
        eata_entropy_weighting=args.eata_entropy_weighting,
        eata_weight_scale=args.eata_weight_scale,
        anchor_reg_weight=args.anchor_reg_weight,
        aug_view_ratio=args.aug_view_ratio,
        entropy_ratio=args.entropy_ratio,
        high_thresh=args.high_thresh,
        low_thresh=args.low_thresh,
        domain_feature_size=args.domain_feature_size,
        fix_ema_teacher=args.fix_ema_teacher,
        deterministic_prediction=args.deterministic_prediction,
        post_update_prediction=args.post_update_prediction,
        new_domain_init=args.new_domain_init,
    )
    LOGGER.info("moe_args=%s", vars(args))
    LOGGER.info("optimizer=%s", optimizer)
    return wrapped


def setup_optimizer_moe(model_params, moe_params, model_lr, moe_lr, optim_cfg):
    groups = [{"params": moe_params, "lr": moe_lr}]
    if model_params:
        groups.append({"params": model_params, "lr": model_lr})
    if optim_cfg.METHOD != "Adam":
        raise ValueError("the clean ImageNet project supports Adam only")
    return optim.Adam(
        groups,
        lr=model_lr,
        betas=(optim_cfg.BETA, 0.999),
        weight_decay=optim_cfg.WD,
    )
