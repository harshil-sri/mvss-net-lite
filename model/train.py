import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# As requested, do not modify these but assume they exist
from model.network import MVSSNetLite
from model.fusion import CBAMFusion
from data_pipeline.dataset_loader import get_dataloader

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
        # Create pos_weight tensor on the same device as inputs
        pos_weight = torch.tensor([self.pos_weight_val]).to(inputs.device)
        
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

parser = argparse.ArgumentParser(description="Train MVSS-Net Lite")
parser.add_argument("--datasets", nargs='+', default=['CASIAv2', 'DEFACTO'], help="Datasets to train on")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
parser.add_argument("--smoke-test", action='store_true', help="Run a quick 2-batch smoke test")
parser.add_argument("--overfit-batch", action='store_true', help="Overfit on a single batch for rapid prototyping")
parser.add_argument("--stage-name", type=str, default="stage1", help="Name prefix for saving plots and models")
parser.add_argument("--init-weights", type=str, default=None, help="Path to checkpoint to initialize weights from")
parser.add_argument("--resume", action='store_true', help="Resume training from init_weights and append to history")
args = parser.parse_args()

DATASETS = args.datasets
EPOCHS = args.epochs
SMOKE_TEST = args.smoke_test
OVERFIT_BATCH = args.overfit_batch
SMOKE_TEST_EPOCHS = 2
SMOKE_TEST_BATCHES = 2

# Bump LR automatically for overfit sanity checks to avoid step-starvation
if OVERFIT_BATCH:
    LEARNING_RATE = 1e-3
    print("OVERFIT MODE: Setting global seed for deterministic split and batch...")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize model
    model = MVSSNetLite().to(device)
    
    # 2. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_epoch = 1
    
    if args.init_weights and os.path.exists(args.init_weights):
        print(f"Loading weights from {args.init_weights}...")
        checkpoint = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if args.resume and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
            
        print("Weights loaded successfully!")
    
    # 3. Loss functions
    # Using Combined BCE + Dice Loss with a heavy pos_weight (50.0) 
    # to severely penalize the model for missing the 1% forged pixels.
    # This prevents the mode collapse where it just predicts all zeros.
    seg_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=50.0)
    edge_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=500.0, use_tversky=True)
    
    # 4. DataLoader
    print(f"Loading datasets: {DATASETS}...")
    train_loader, val_loader, test_loader = get_dataloader(DATASETS, batch_size=BATCH_SIZE, is_train=True, return_splits=True)
    print(f"Dataset splits -> Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")
    
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
        for imgs, masks, edges in train_loader:
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

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        
        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        epoch_seg_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_total_loss = 0.0
        batches_processed = 0
        
        if OVERFIT_BATCH:
            batch_iterator = [(overfit_imgs, overfit_masks, overfit_edges)]
        else:
            batch_iterator = train_loader
            
        for batch_idx, (imgs, masks, edges) in enumerate(batch_iterator):
            if SMOKE_TEST and batches_processed >= SMOKE_TEST_BATCHES:
                break
                
            if not OVERFIT_BATCH:
                # Move to device
                imgs = imgs.to(device)
                masks = masks.to(device)
                edges = edges.to(device)
            
            # Zero grads
            optimizer.zero_grad()
            
            # Forward pass
            # MVSSNetLite forward returns (seg_mask, edge_map)
            pred_seg, pred_edge = model(imgs)
            
            # Compute losses
            loss_seg = seg_criterion(pred_seg, masks)
            loss_edge = edge_criterion(pred_edge, edges)
            
            loss_total = (SEG_LOSS_WEIGHT * loss_seg) + (EDGE_LOSS_WEIGHT * loss_edge)
            
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
        
        print(f"=== Epoch {epoch} Summary ===")
        print(f"Time: {epoch_duration:.1f}s | LR: {current_lr:.6f} | GPU Mem: {gpu_mem:.0f} MB")
        print(f"TRAIN -> Avg Seg: {avg_seg:.4f} | Avg Edge: {avg_edge:.4f} | Avg Total: {avg_total:.4f}")
        print(f"VAL   -> Avg Seg: {avg_val_seg:.4f} | Avg Edge: {avg_val_edge:.4f} | Avg Total: {avg_val_total:.4f}\n")
        
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
        
        # Save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == total_epochs:
            chkpt_path = f"model/checkpoints/{args.stage_name}_mvss_lite_ep{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total
            }, chkpt_path)
            print(f"Checkpoint saved to {chkpt_path}")
            
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
    
    # Save statistics to CSV
    csv_path = f"reports/{args.stage_name}_history.csv"
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_seg_loss', 'train_edge_loss', 'train_total_loss', 'val_seg_loss', 'val_edge_loss', 'val_total_loss', 'learning_rate', 'epoch_time_sec', 'gpu_mem_mb'])
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i], 
                history['seg_loss'][i], 
                history['edge_loss'][i], 
                history['total_loss'][i],
                history['val_seg_loss'][i],
                history['val_edge_loss'][i],
                history['val_total_loss'][i],
                history['learning_rate'][i],
                history['epoch_time_sec'][i],
                history['gpu_mem_mb'][i]
            ])
    print(f"Statistics saved to {csv_path}")


if __name__ == '__main__':
    train()
