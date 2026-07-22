import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# As requested, do not modify these but assume they exist
from model.network import MVSSNetLite
from model.fusion import CBAMFusion
from data_pipeline.dataset_loader import get_dataloader

import argparse

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
parser.add_argument("--stage-name", type=str, default="stage1", help="Name prefix for saving plots and models")
parser.add_argument("--init-weights", type=str, default=None, help="Path to checkpoint to initialize weights from")
parser.add_argument("--resume", action='store_true', help="Resume training from init_weights and append to history")
args = parser.parse_args()

DATASETS = args.datasets
EPOCHS = args.epochs
SMOKE_TEST = args.smoke_test
SMOKE_TEST_EPOCHS = 2
SMOKE_TEST_BATCHES = 2


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
    # Using BCEWithLogitsLoss because usually networks output raw logits 
    # instead of post-sigmoid probabilities for numerical stability.
    seg_criterion = nn.BCEWithLogitsLoss()
    edge_criterion = nn.BCEWithLogitsLoss()
    
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
        'val_total_loss': []
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
            print(f"Loaded existing history from {csv_path}")
    
    print("Starting training loop...")
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        
        epoch_seg_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_total_loss = 0.0
        batches_processed = 0
        
        for batch_idx, (imgs, masks, edges) in enumerate(train_loader):
            if SMOKE_TEST and batches_processed >= SMOKE_TEST_BATCHES:
                break
                
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
            for v_imgs, v_masks, v_edges in val_loader:
                if SMOKE_TEST and val_batches >= SMOKE_TEST_BATCHES:
                    break
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
        
        print(f"=== Epoch {epoch} Summary ===")
        print(f"TRAIN -> Avg Seg: {avg_seg:.4f} | Avg Edge: {avg_edge:.4f} | Avg Total: {avg_total:.4f}")
        print(f"VAL   -> Avg Seg: {avg_val_seg:.4f} | Avg Edge: {avg_val_edge:.4f} | Avg Total: {avg_val_total:.4f}\n")
        
        history['epoch'].append(epoch)
        history['seg_loss'].append(avg_seg)
        history['edge_loss'].append(avg_edge)
        history['total_loss'].append(avg_total)
        
        history['val_seg_loss'].append(avg_val_seg)
        history['val_edge_loss'].append(avg_val_edge)
        history['val_total_loss'].append(avg_val_total)
        
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
        writer.writerow(['epoch', 'train_seg_loss', 'train_edge_loss', 'train_total_loss', 'val_seg_loss', 'val_edge_loss', 'val_total_loss'])
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i], 
                history['seg_loss'][i], 
                history['edge_loss'][i], 
                history['total_loss'][i],
                history['val_seg_loss'][i],
                history['val_edge_loss'][i],
                history['val_total_loss'][i]
            ])
    print(f"Statistics saved to {csv_path}")


if __name__ == '__main__':
    train()
