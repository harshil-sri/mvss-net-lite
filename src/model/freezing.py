"""
Layer freezing for Stage 2 Run 2 (prd.md Section 3).

Strategy: freeze the early stages of BOTH ResNet-34 backbones (RGB/edge branch
and Bayar-constrained noise branch) - stem conv + first two ResNet stages -
and fine-tune only the later stages + all fusion/decoder/head modules.

Per PRD Section 3.2:
- requires_grad=False on frozen params AND optimizer must be built only over
  trainable params (filter(lambda p: p.requires_grad, ...)).
- BatchNorm layers inside frozen stages must be kept in eval() mode so their
  running stats do not drift on the new domain while conv weights stay fixed.
"""

import torch.nn as nn


def get_frozen_module_groups(model):
    """
    Returns an ordered dict: group name -> list of modules to freeze.
    Noise branch includes the Bayar-constrained conv itself (PRD 3.1).
    """
    bb = model.backbone
    return {
        "noise_branch[stem+L1+L2+bayar]": [
            bb.noise_extractor,
            bb.noise_conv1,
            bb.noise_bn1,
            bb.noise_layer1,
            bb.noise_layer2,
        ],
        "edge_branch[stem+L1+L2]": [
            bb.edge_conv1,
            bb.edge_bn1,
            bb.edge_layer1,
            bb.edge_layer2,
        ],
    }


def apply_stage2_freeze(model):
    """
    Sets requires_grad=False on all parameters of the frozen groups.
    Returns the frozen module groups dict (for logging / BN collection).
    Idempotent: safe to call multiple times.
    """
    groups = get_frozen_module_groups(model)
    for mods in groups.values():
        for m in mods:
            for p in m.parameters():
                p.requires_grad = False
    return groups


def collect_frozen_bn_layers(frozen_groups):
    """
    All BatchNorm modules living anywhere inside the frozen groups.
    Accepts either the groups dict returned by apply_stage2_freeze() or the
    model itself (in which case the groups are derived from it).
    """
    if isinstance(frozen_groups, nn.Module):
        frozen_groups = get_frozen_module_groups(frozen_groups)
    bns = []
    seen = set()
    for mods in frozen_groups.values():
        for m in mods:
            for sub in m.modules():
                if isinstance(sub, nn.modules.batchnorm._BatchNorm) and id(sub) not in seen:
                    seen.add(id(sub))
                    bns.append(sub)
    return bns


def keep_frozen_bn_eval(bn_layers):
    """
    Call AFTER every model.train() so frozen BatchNorm running stats stay
    frozen (eval mode) while the rest of the network trains.
    """
    for bn in bn_layers:
        bn.eval()


def count_params(module_or_params):
    if isinstance(module_or_params, nn.Module):
        params = module_or_params.parameters()
    else:
        params = module_or_params
    return sum(p.numel() for p in params)


def freeze_report(model, frozen_groups):
    """
    Returns a dict of raw parameter counts for startup logging:
      per-group frozen counts, per-module trainable counts, totals.
    No verdicts - just numbers.
    """
    report = {}
    total_all = count_params(model)
    frozen_total = 0

    for group_name, mods in frozen_groups.items():
        gcount = sum(count_params(m) for m in mods)
        frozen_total += gcount
        report[f"frozen::{group_name}"] = gcount

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Trainable breakdown by top-level component (numbers only, no overlap:
    # backbone vs fusion vs decoder vs heads)
    bb_trainable = sum(p.numel() for n, p in model.backbone.named_parameters() if p.requires_grad)
    fuse_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and ".fuse" in name and name.startswith("backbone.")
    )
    decoder_heads_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("backbone.")
    )

    report["trainable::backbone[layer3+layer4 both branches]"] = bb_trainable - fuse_trainable
    report["trainable::cbam_fusion[fuse1-4]"] = fuse_trainable
    report["trainable::decoder+heads"] = decoder_heads_trainable
    report["TOTAL::trainable"] = trainable_total
    report["TOTAL::frozen"] = frozen_total
    report["TOTAL::all"] = total_all

    # Sanity arithmetic (report raw, let reviewer reconcile)
    report["CHECK::trainable+frozen==all"] = (trainable_total + frozen_total) == total_all
    return report


def format_freeze_report(report):
    lines = ["Freeze report (parameter counts):"]
    for k, v in report.items():
        if isinstance(v, bool):
            lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {k}: {v:,}")
    return "\n".join(lines)
