import math
import torch.nn.functional as F

import os
import shutil
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# As requested, do not modify these but assume they exist
from model.network import MVSSNetLite
from model.fusion import CBAMFusion
from data_pipeline.dataset_loader import get_dataloader
from model.freezing import (
    apply_stage2_freeze,
    collect_frozen_bn_layers,
    keep_frozen_bn_eval,
    freeze_report,
    format_freeze_report,
)

import argparse
import random
import numpy as np
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight_val=50.0, use_tversky=False):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight_val = pos_weight_val
        self.use_tversky = use_tversky

    def forward(self, inputs, targets):
        pos_weight = torch.tensor([self.pos_weight_val], device=inputs.device)

        # Calculate BCE loss with pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)

        # Calculate Region loss (expects probabilities)
        probs = torch.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        smooth = 1e-6
        if self.use_tversky:
            alpha = 0.3
            beta = 0.7
            intersection = (probs_flat * targets_flat).sum()
            fps = (probs_flat * (1 - targets_flat)).sum()
            fns = ((1 - probs_flat) * targets_flat).sum()
            region_loss = 1 - ((intersection + smooth) / (intersection + alpha * fps + beta * fns + smooth))
        else:
            intersection = (probs_flat * targets_flat).sum()
            region_loss = 1 - ((2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth))

        return (self.bce_weight * bce_loss) + (self.dice_weight * region_loss)

# =============================================================================
# HYPERPARAMETERS & CONFIG
# =============================================================================
SEG_LOSS_WEIGHT = 1.0
EDGE_LOSS_WEIGHT = 1.0

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
SAVE_EVERY = 5
SMOKE_TEST_EPOCHS = 2
SMOKE_TEST_BATCHES = 20

# Stage 2 Run 2 (prd.md Section 2.2): realized sampler composition must be
# logged to a persistent file from the very first batch, at minimum every 50
# batches.
LOG_SAMPLER_EVERY = 50
MIN_FREE_GB_BEFORE_SAVE = 15.0


def save_checkpoint_with_guard(payload, chkpt_path, min_free_gb=MIN_FREE_GB_BEFORE_SAVE,
                               warning_dir='reports', context_msg=''):
    """
    Saves a checkpoint only if enough free disk space exists.
    Returns (saved: bool, free_space_gb: float).
    On insufficient space: writes a persistent warning file and does NOT save
    and does NOT raise - the caller decides to halt gracefully.
    """
    free_space_gb = shutil.disk_usage("/").free / (1024**3)
    if free_space_gb < min_free_gb:
        os.makedirs(warning_dir, exist_ok=True)
        msg = (f"Halted before saving {chkpt_path}: free space {free_space_gb:.2f}GB "
               f"< required {min_free_gb}GB. {context_msg}")
        with open(os.path.join(warning_dir, "DISK_FULL_WARNING.txt"), "w") as f:
            f.write(msg + "\n")
        return False, free_space_gb
    torch.save(payload, chkpt_path)
    return True, free_space_gb


def rotate_checkpoints(saved_checkpoints, keep=3):
    """
    Keeps only the most recent `keep` checkpoints. Mutates saved_checkpoints in place.
    Returns the list of paths actually deleted.
    """
    if len(saved_checkpoints) <= keep:
        return []

    removed = []
    candidates = saved_checkpoints[:-keep].copy()
    for c in candidates:
        if os.path.exists(c):
            os.remove(c)
        removed.append(c)
        saved_checkpoints.remove(c)

    return removed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MVSS-Net Lite")
    parser.add_argument("--datasets", nargs='+', default=['CASIAv2', 'DEFACTO'], help="Datasets to train on")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--smoke-test", action='store_true', help="Run a quick 2-batch smoke test")
    parser.add_argument("--overfit-batch", action='store_true', help="Overfit on a single batch for rapid prototyping")
    parser.add_argument("--stage-name", type=str, default="stage1", help="Name prefix for saving plots and models")
    parser.add_argument("--init-weights", type=str, default=None, help="Path to checkpoint to initialize weights from")
    parser.add_argument("--resume", action='store_true', help="Resume training from init_weights and append to history")
    parser.add_argument("--use-balanced-sampler", action='store_true', help="Enable WeightedRandomSampler for dataset balancing")
    parser.add_argument('--lr-schedule', type=str, default='flat', choices=['flat', 'cosine'], help='Learning rate schedule')
    parser.add_argument('--early-stopping-patience', type=int, default=-1, help='Epochs to wait before early stop (-1 to disable)')
    parser.add_argument('--domain-adversarial', action='store_true', help='Use domain adversarial loss')
    parser.add_argument("--freeze-early-layers", action='store_true',
                        help="Stage 2 Run 2 experiment: freeze stem+layer1+layer2 of both backbones "
                             "(incl. Bayar constrained conv); train layer3/layer4 + fusion + decoder + heads")
    return parser.parse_args()


def log_sampler_composition(log_path, epoch, batch_idx, idx_list, classes):
    """Writes one raw-composition line to stdout and the persistent log file."""
    n = len(idx_list)
    counts = {'rtm_forged': 0, 'rtm_auth': 0, 'midv500': 0}
    other = 0
    for i in idx_list:
        c = classes[i]
        if c in counts:
            counts[c] += 1
        else:
            other += 1
    line = (f"[sampler] epoch={epoch} batch={batch_idx} n={n} "
            f"rtm_forged={counts['rtm_forged']} ({100.0*counts['rtm_forged']/n:.2f}%) "
            f"rtm_authentic={counts['rtm_auth']} ({100.0*counts['rtm_auth']/n:.2f}%) "
            f"midv500={counts['midv500']} ({100.0*counts['midv500']/n:.2f}%) "
            f"other={other}")
    print(line)
    with open(log_path, 'a') as f:
        f.write(line + "\n")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    SMOKE_TEST = args.smoke_test
    OVERFIT_BATCH = args.overfit_batch

    # Bump LR automatically for overfit sanity checks to avoid step-starvation
    if OVERFIT_BATCH:
        LEARNING_RATE = 1e-3
        print("OVERFIT MODE: Setting global seed for deterministic split and batch...")
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

    DATASETS = args.datasets

    # 1. Initialize model
    model = MVSSNetLite().to(device)

    start_epoch = 1

    if args.init_weights and os.path.exists(args.init_weights):
        print(f"Loading weights from {args.init_weights}...")
        checkpoint = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if args.resume and 'optimizer_state_dict' in checkpoint:
            # Optimizer is constructed below over the (possibly frozen) trainable
            # subset; state dict loading happens right after its construction.
            pass

        print("Weights loaded successfully!")
    elif args.init_weights:
        print(f"WARNING: init-weights path {args.init_weights} does not exist; starting fresh.")

    # 2. Optional layer freezing (Stage 2 Run 2 experiment) - MUST happen
    #    before optimizer construction so the optimizer only ever sees
    #    trainable parameters (PRD Section 3.2).
    frozen_bn_layers = []
    if args.freeze_early_layers:
        frozen_groups = apply_stage2_freeze(model)
        frozen_bn_layers = collect_frozen_bn_layers(model)
        print(format_freeze_report(freeze_report(model, frozen_groups)))
        print(f"Frozen BatchNorm layers kept in eval mode: {len(frozen_bn_layers)}")

    # 3. Optimizer - over trainable params only when freezing is active
    if args.freeze_early_layers:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Optimizer built over TRAINABLE params only: {len(trainable_params)} tensors")
        optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if args.resume and args.init_weights and os.path.exists(args.init_weights):
        checkpoint = torch.load(args.init_weights, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")

    # 4. DataLoader
    # return_index=True whenever the balanced sampler is on, so realized
    # per-batch composition can be logged from batch one (PRD Section 2.2).
    print(f"Loading datasets: {DATASETS}...")
    train_loader, val_loader, test_loader = get_dataloader(
        DATASETS, batch_size=BATCH_SIZE, is_train=True, return_splits=True,
        use_balanced_sampler=args.use_balanced_sampler,
        return_index=args.use_balanced_sampler)
    print(f"Dataset splits -> Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")

    sampler_comp_log = None
    sampler_classes = None
    indexed_batches = args.use_balanced_sampler
    if args.use_balanced_sampler:
        sampler_comp_log = f"reports/{args.stage_name}_sampler_composition.log"
        ds = train_loader.dataset
        sampler_classes = getattr(ds.base, 'sample_classes', None) if hasattr(ds, 'base') else getattr(ds, 'sample_classes', None)
        if sampler_classes is None:
            indexed_batches = False
            print("WARNING: could not locate sample_classes; sampler composition logging DISABLED.")
        else:
            print(f"Sampler composition logging -> {sampler_comp_log} (every {LOG_SAMPLER_EVERY} batches, from batch 1)")

    # 5. Loss functions & Dataset Scanner
    print(f"Scanning dataset {DATASETS} to compute global pixel statistics (this may take a minute)...")
    total_mask_pos, total_mask_neg = 0.0, 0.0
    total_edge_pos, total_edge_neg = 0.0, 0.0

    # We only scan if not overfitting to a single batch, otherwise just use fallbacks
    if OVERFIT_BATCH:
        seg_pos_weight = 124.0
        edge_pos_weight = 2495.0
    else:
        scan_iter = train_loader
        if indexed_batches:
            for i, (imgs, masks, edges, _idx) in enumerate(scan_iter):
                total_mask_pos += masks.sum().item()
                total_mask_neg += (masks.numel() - masks.sum().item())
                total_edge_pos += edges.sum().item()
                total_edge_neg += (edges.numel() - edges.sum().item())
                if SMOKE_TEST and i > 2:
                    break
        else:
            for i, (imgs, masks, edges) in enumerate(scan_iter):
                total_mask_pos += masks.sum().item()
                total_mask_neg += (masks.numel() - masks.sum().item())
                total_edge_pos += edges.sum().item()
                total_edge_neg += (edges.numel() - edges.sum().item())
                if SMOKE_TEST and i > 2:
                    break

        seg_pos_weight = total_mask_neg / max(total_mask_pos, 1.0)
        edge_pos_weight = total_edge_neg / max(total_edge_pos, 1.0)

    print(f"Global Stats -> Seg pos_weight: {seg_pos_weight:.2f} | Edge pos_weight: {edge_pos_weight:.2f}")

    seg_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=seg_pos_weight)
    edge_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=edge_pos_weight, use_tversky=True)

    # Ensure output dirs exist
    os.makedirs('model/checkpoints', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    total_epochs = SMOKE_TEST_EPOCHS if SMOKE_TEST else EPOCHS

    # Tracking for plot
    history = {
        'epoch': [],
        'seg_loss': [],
        'edge_loss': [],
        'total_loss': [],
        'val_seg_loss': [],
        'val_edge_loss': [],
        'val_total_loss': [],
        'learning_rate': [],
        'epoch_time_sec': [],
        'gpu_mem_mb': []
    }

    if args.resume:
        csv_path = f"reports/{args.stage_name}_history.csv"
        if os.path.exists(csv_path):
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    history['epoch'].append(int(row['epoch']))
                    history['seg_loss'].append(float(row['train_seg_loss']))
                    history['edge_loss'].append(float(row['train_edge_loss']))
                    history['total_loss'].append(float(row['train_total_loss']))
                    history['val_seg_loss'].append(float(row['val_seg_loss']))
                    history['val_edge_loss'].append(float(row['val_edge_loss']))
                    history['val_total_loss'].append(float(row['val_total_loss']))

                    # Handle backwards compatibility if old CSV lacks new metrics
                    history['learning_rate'].append(float(row.get('learning_rate', LEARNING_RATE)))
                    history['epoch_time_sec'].append(float(row.get('epoch_time_sec', 0.0)))
                    history['gpu_mem_mb'].append(float(row.get('gpu_mem_mb', 0.0)))
            print(f"Loaded existing history from {csv_path}")

    print("Starting training loop...")

    if OVERFIT_BATCH:
        print("OVERFIT MODE: Grabbing a single batch of 8 perfectly manipulated images...")
        manip_imgs, manip_masks, manip_edges = [], [], []
        for batch_data in train_loader:
            if indexed_batches:
                imgs, masks, edges = batch_data[0], batch_data[1], batch_data[2]
            else:
                imgs, masks, edges = batch_data
            for i in range(imgs.size(0)):
                if masks[i].sum() > 0:
                    manip_imgs.append(imgs[i])
                    manip_masks.append(masks[i])
                    manip_edges.append(edges[i])
                if len(manip_imgs) == 8:
                    break
            if len(manip_imgs) == 8:
                break

        overfit_imgs = torch.stack(manip_imgs)
        overfit_masks = torch.stack(manip_masks)
        overfit_edges = torch.stack(manip_edges)
        overfit_imgs = overfit_imgs.to(device)
        overfit_masks = overfit_masks.to(device)
        overfit_edges = overfit_edges.to(device)

        # Save these exact images and masks so the user can test them
        os.makedirs("reports/overfit_samples", exist_ok=True)
        import torchvision
        for i in range(overfit_imgs.size(0)):
            # Dataloader outputs in [0, 1] range, no ImageNet inv_normalize needed
            torchvision.utils.save_image(overfit_imgs[i].cpu(), f"reports/overfit_samples/overfit_{i}.jpg")
            torchvision.utils.save_image(overfit_masks[i].cpu().float(), f"reports/overfit_samples/overfit_{i}_gt.png")

    saved_checkpoints = []

    halted_low_disk = False
    
    # Setup LR Scheduler
    scheduler = None
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    # Setup early stopping
    es_patience = args.early_stopping_patience
    es_counter = 0
    es_best_loss = float('inf')
    es_stop = False
    
    # Setup best val logic
    global_best_val_loss = float('inf')
    best_val_checkpoint_path = None
    
    # Pre-compute total batches for lambda schedule
    total_steps = len(train_loader) * total_epochs
    global_step = 0
    
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        # Frozen BatchNorm layers must NOT update running stats on the new
        # domain while their conv weights are fixed (PRD Section 3.2).
        if frozen_bn_layers:
            keep_frozen_bn_eval(frozen_bn_layers)

        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        epoch_seg_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_total_loss = 0.0
        batches_processed = 0
        epoch_domain_correct = 0
        epoch_domain_total = 0

        if OVERFIT_BATCH:
            batch_iterator = [(overfit_imgs, overfit_masks, overfit_edges)]
        else:
            batch_iterator = train_loader

        for batch_idx, batch_data in enumerate(batch_iterator):
            if SMOKE_TEST and batches_processed >= SMOKE_TEST_BATCHES:
                break

            if indexed_batches and not OVERFIT_BATCH:
                imgs, masks, edges, sample_idxs = batch_data
            else:
                imgs, masks, edges = batch_data[0], batch_data[1], batch_data[2]

            if not OVERFIT_BATCH:
                # Move to device
                imgs = imgs.to(device)
                masks = masks.to(device)
                edges = edges.to(device)

            # Realized sampler composition logging (PRD Section 2.2):
            # raw per-batch counts from batch one, every LOG_SAMPLER_EVERY batches.
            if sampler_classes is not None and not OVERFIT_BATCH:
                if batches_processed == 0 or (batches_processed + 1) % LOG_SAMPLER_EVERY == 0 or batch_idx == 0:
                    log_sampler_composition(
                        sampler_comp_log, epoch, batches_processed + 1,
                        sample_idxs.tolist(), sampler_classes)

            # Zero grads
            optimizer.zero_grad()

            # GRL lambda calculation
            lambda_p = None
            if args.domain_adversarial:
                p = global_step / float(total_steps)
                lambda_p = (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0

            # Forward pass
            if args.domain_adversarial:
                pred_seg, pred_edge, domain_logits = model(imgs, lambda_p=lambda_p)
            else:
                pred_seg, pred_edge = model(imgs)

            # Compute losses
            loss_seg = seg_criterion(pred_seg, masks)
            loss_edge = edge_criterion(pred_edge, edges)
            
            domain_loss_t = 0.0
            domain_correct = 0
            domain_total = 0
            if args.domain_adversarial and sampler_classes is not None:
                labels_str = [sampler_classes[idx] for idx in sample_idxs.tolist()]
                valid_mask = torch.tensor([l in ('rtm_forged', 'rtm_auth', 'midv500') for l in labels_str], device=device)
                if valid_mask.sum() > 0:
                    domain_targets = torch.tensor([1 if l == 'midv500' else 0 for l in labels_str], device=device)
                    domain_logits_valid = domain_logits[valid_mask]
                    domain_targets_valid = domain_targets[valid_mask]
                    
                    domain_loss_t = F.cross_entropy(domain_logits_valid, domain_targets_valid)
                    
                    domain_preds = torch.argmax(domain_logits_valid, dim=1)
                    domain_correct += (domain_preds == domain_targets_valid).sum().item()
                    domain_total += len(domain_targets_valid)

            loss_total = (SEG_LOSS_WEIGHT * loss_seg) + (EDGE_LOSS_WEIGHT * loss_edge)
            if args.domain_adversarial:
                loss_total += 0.1 * domain_loss_t

            # Backward pass & step
            loss_total.backward()
            optimizer.step()

            # Normalize the constrained conv layer weights!
            model.backbone.noise_extractor.normalize_weights()

            # Accumulate
            epoch_seg_loss += loss_seg.item()
            epoch_edge_loss += loss_edge.item()
            epoch_total_loss += loss_total.item()
            batches_processed += 1
            global_step += 1
            if args.domain_adversarial and sampler_classes is not None and domain_total > 0:
                epoch_domain_correct += domain_correct
                epoch_domain_total += domain_total

            if batches_processed % 10 == 0 or SMOKE_TEST:
                print(f"Epoch [{epoch}/{total_epochs}] Batch [{batches_processed}] - "
                      f"Seg Loss: {loss_seg.item():.4f}, "
                      f"Edge Loss: {loss_edge.item():.4f}, "
                      f"Total Loss: {loss_total.item():.4f}")

        # Calculate averages for the epoch
        avg_seg = epoch_seg_loss / max(1, batches_processed)
        avg_edge = epoch_edge_loss / max(1, batches_processed)
        avg_total = epoch_total_loss / max(1, batches_processed)

        # --- VALIDATION LOOP ---
        model.eval()
        val_seg_loss, val_edge_loss, val_total_loss = 0.0, 0.0, 0.0
        val_batches = 0
        with torch.no_grad():
            if OVERFIT_BATCH:
                val_iterator = [(overfit_imgs, overfit_masks, overfit_edges)]
            else:
                val_iterator = val_loader

            for v_imgs, v_masks, v_edges in val_iterator:
                if SMOKE_TEST and val_batches >= SMOKE_TEST_BATCHES:
                    break
                if not OVERFIT_BATCH:
                    v_imgs, v_masks, v_edges = v_imgs.to(device), v_masks.to(device), v_edges.to(device)
                if args.domain_adversarial:
                    v_pred_seg, v_pred_edge, _ = model(v_imgs, lambda_p=0.0)
                else:
                    v_pred_seg, v_pred_edge = model(v_imgs)
                vl_seg = seg_criterion(v_pred_seg, v_masks)
                vl_edge = edge_criterion(v_pred_edge, v_edges)
                vl_total = (SEG_LOSS_WEIGHT * vl_seg) + (EDGE_LOSS_WEIGHT * vl_edge)
                val_seg_loss += vl_seg.item()
                val_edge_loss += vl_edge.item()
                val_total_loss += vl_total.item()
                val_batches += 1

        avg_val_seg = val_seg_loss / max(1, val_batches)
        avg_val_edge = val_edge_loss / max(1, val_batches)
        avg_val_total = val_total_loss / max(1, val_batches)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        gpu_mem = torch.cuda.max_memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0.0


        domain_acc = epoch_domain_correct / max(1, epoch_domain_total) if epoch_domain_total > 0 else 0.0
        if scheduler is not None:
            scheduler.step()

        print(f"=== Epoch {epoch} Summary ===")
        print(f"Time: {epoch_duration:.1f}s | LR: {current_lr:.6f} | GPU Mem: {gpu_mem:.0f} MB")
        if args.domain_adversarial:
            print(f"TRAIN -> Avg Seg: {avg_seg:.4f} | Avg Edge: {avg_edge:.4f} | Avg Total: {avg_total:.4f} | Domain Acc: {domain_acc*100:.2f}%")
        else:
            print(f"TRAIN -> Avg Seg: {avg_seg:.4f} | Avg Edge: {avg_edge:.4f} | Avg Total: {avg_total:.4f}")
        print(f"VAL   -> Avg Seg: {avg_val_seg:.4f} | Avg Edge: {avg_val_edge:.4f} | Avg Total: {avg_val_total:.4f}\n")


        import csv
        csv_path = f"reports/{args.stage_name}_history.csv"
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_seg_loss', 'train_edge_loss', 'train_total_loss', 'val_seg_loss', 'val_edge_loss', 'val_total_loss', 'learning_rate', 'epoch_time_sec', 'gpu_mem_mb'])
            writer.writerow([
                epoch, avg_seg, avg_edge, avg_total,
                avg_val_seg, avg_val_edge, avg_val_total,
                optimizer.param_groups[0]['lr'], epoch_duration, gpu_mem
            ])

        history['epoch'].append(epoch)
        history['seg_loss'].append(avg_seg)
        history['edge_loss'].append(avg_edge)
        history['total_loss'].append(avg_total)

        history['val_seg_loss'].append(avg_val_seg)
        history['val_edge_loss'].append(avg_val_edge)
        history['val_total_loss'].append(avg_val_total)

        history['learning_rate'].append(current_lr)
        history['epoch_time_sec'].append(epoch_duration)
        history['gpu_mem_mb'].append(gpu_mem)

        # Check best validation loss and save _bestval checkpoint
        if avg_val_total < global_best_val_loss:
            global_best_val_loss = avg_val_total
            bestval_path = f"model/checkpoints/{args.stage_name}_mvss_lite_ep{epoch}_bestval.pt"
            
            payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_total
            }
            torch.save(payload, bestval_path)
            print(f"*** New best validation total loss: {avg_val_total:.4f} at epoch {epoch}. Saved {bestval_path} ***")
            
            if best_val_checkpoint_path and best_val_checkpoint_path != bestval_path and os.path.exists(best_val_checkpoint_path):
                os.remove(best_val_checkpoint_path)
            best_val_checkpoint_path = bestval_path
            
            es_counter = 0
        else:
            es_counter += 1
            print(f"Early stopping counter: {es_counter} out of {es_patience}")
            
        if es_patience > 0 and es_counter >= es_patience:
            es_stop = True

        # Save checkpoint (selectively, with rotation + disk-space guardrail)
        if epoch % SAVE_EVERY == 0 or epoch == total_epochs:
            chkpt_path = f"model/checkpoints/{args.stage_name}_mvss_lite_ep{epoch}.pt"
            payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total
            }
            ok, free_space_gb = save_checkpoint_with_guard(
                payload, chkpt_path,
                context_msg=f"(stage={args.stage_name}, epoch={epoch})")

            if not ok:
                print(f"CRITICAL WARNING: Free disk space dropped to {free_space_gb:.2f}GB! Halting training to prevent crash.")
                halted_low_disk = True
                break

            print(f"Checkpoint saved to {chkpt_path} (free space after save: {free_space_gb:.2f}GB)")

            saved_checkpoints.append(chkpt_path)
            # Do not pass best_checkpoint_path to rotation, it's handled separately
            removed = rotate_checkpoints(saved_checkpoints)
            for old_chkpt in removed:
                print(f"Deleted old checkpoint {old_chkpt}")


        if es_stop:
            print(f"Early stopping triggered at epoch {epoch}. Halting training.")
            break

    # Plotting the loss curve
    print("Generating loss curve plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['seg_loss'], label='Train Seg Loss', marker='o')
    plt.plot(history['epoch'], history['edge_loss'], label='Train Edge Loss', marker='o')
    plt.plot(history['epoch'], history['total_loss'], label='Train Total Loss', marker='o', linewidth=2)
    plt.plot(history['epoch'], history['val_total_loss'], label='Val Total Loss', marker='s', linestyle='--', linewidth=2)

    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = f"reports/{args.stage_name}_loss_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Training finished! Plot saved to {plot_path}")

    if halted_low_disk:
        print("NOTE: Training halted early due to low disk space (see reports/DISK_FULL_WARNING.txt).")

    print("Done!")


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
