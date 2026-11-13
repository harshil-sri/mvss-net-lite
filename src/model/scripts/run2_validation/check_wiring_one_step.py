"""
Check 6: single forward+backward wiring check. ONE batch, ONE step, no
scheduling, no epoch loop - this proves model + losses + freezing + new
dataloader are wired together correctly end to end.

Assertions (raw numbers printed):
  - stage1 checkpoint loads into MVSSNetLite cleanly
  - freeze applied: per-branch parameter counts printed, arithmetic reconciles
  - losses are finite
  - after backward: zero grads on FROZEN params, nonzero grads present on
    TRAINABLE params
  - after one optimizer.step() + normalize_weights(): Bayar constraint intact
    (center weight == -1, surrounding weights sum to +1, per filter)

Run from repo root:
    python -m model.scripts.run2_validation.check_wiring_one_step            # freeze ON (Run 2 recipe)
    python -m model.scripts.run2_validation.check_wiring_one_step --no-freeze  # plain pipeline
"""
import sys

import torch

from model.network import MVSSNetLite
from model.train import CombinedLoss
from data_pipeline.dataset_loader import get_dataloader
from model.freezing import (
    apply_stage2_freeze, collect_frozen_bn_layers, keep_frozen_bn_eval,
    freeze_report, format_freeze_report,
)

INIT_CKPT = 'model/checkpoints/stage1_mvss_lite_ep45.pt'
BATCH_SIZE = 8


def main():
    FREEZE = '--no-freeze' not in sys.argv
    print(f"freeze mode: {'ON (stem+L1+L2 both branches)' if FREEZE else 'OFF (full network trainable)'}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # 1. Model + fresh init from Stage 1 ep45 (the Run 2 recipe)
    model = MVSSNetLite().to(device)
    ckpt = torch.load(INIT_CKPT, map_location=device)
    missing_unexpected = model.load_state_dict(ckpt['model_state_dict'])
    print(f"loaded {INIT_CKPT} (epoch={ckpt['epoch']}) | "
          f"missing_keys={len(missing_unexpected.missing_keys)} "
          f"unexpected_keys={len(missing_unexpected.unexpected_keys)}")

    # 2. Optional freezing - ON by default (Run 2 recipe), OFF with --no-freeze
    frozen_bns = []
    if FREEZE:
        groups = apply_stage2_freeze(model)
        report = freeze_report(model, groups)
        print(format_freeze_report(report))
        frozen_bns = collect_frozen_bn_layers(model)

    # 3. Optimizer over trainable-only when frozen; full params otherwise (PRD 3.2)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-4)
    print(f"optimizer tensors: {len(trainable)}")

    # 4. ONE real batch through the manifest loader (same call shape as Run 2)
    train_loader, _, _ = get_dataloader(
        ['RTM', 'MIDV500'], batch_size=BATCH_SIZE, is_train=True,
        return_splits=True, use_balanced_sampler=True, return_index=True)
    batch = next(iter(train_loader))
    imgs, masks, edges, idxs = batch
    imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
    print(f"batch: imgs={tuple(imgs.shape)} masks={tuple(masks.shape)} edges={tuple(edges.shape)} "
          f"idxs={idxs.tolist()}")

    seg_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=222.60)
    edge_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=929.22,
                                  use_tversky=True)

    model.train()
    keep_frozen_bn_eval(frozen_bns)

    pred_seg, pred_edge = model(imgs)
    loss_seg = seg_criterion(pred_seg, masks)
    loss_edge = edge_criterion(pred_edge, edges)
    loss_total = loss_seg + loss_edge
    print(f"loss_seg={loss_seg.item():.4f} loss_edge={loss_edge.item():.4f} "
          f"loss_total={loss_total.item():.4f}")
    finite = torch.isfinite(loss_total).item()

    optimizer.zero_grad()
    loss_total.backward()

    n_frozen_with_grad = sum(1 for p in model.parameters()
                             if not p.requires_grad and p.grad is not None)
    trainable_with_grad = sum(1 for p in trainable if p.grad is not None and p.grad.abs().sum() > 0)
    trainable_total = len(trainable)
    print(f"frozen params that somehow got grads : {n_frozen_with_grad} (must be 0)")
    print(f"trainable params with nonzero grads  : {trainable_with_grad}/{trainable_total}")

    optimizer.step()
    model.backbone.noise_extractor.normalize_weights()

    w = model.backbone.noise_extractor.constrained_conv.weight.data
    centers = w[:, :, 2, 2]
    totals = w.sum(dim=(2, 3))
    center_dev = (centers + 1.0).abs().max().item()
    total_dev = totals.abs().max().item()  # surround=+1 & center=-1 -> total must be ~0
    print(f"bayar constraint: max|center-(-1)|={center_dev:.2e} max|kernel_sum|={total_dev:.2e}")

    ok = (finite
          and n_frozen_with_grad == 0
          and trainable_with_grad == trainable_total
          and center_dev < 1e-4 and total_dev < 1e-3)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == '__main__':
    main()
