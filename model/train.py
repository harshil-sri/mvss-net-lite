import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# As requested, do not modify these but assume they exist
from model.network import MVSSNetLite
from model.fusion import CBAMFusion
from data_pipeline.dataset_loader import get_dataloader

# =============================================================================
# HYPERPARAMETERS & CONFIG
# =============================================================================
SEG_LOSS_WEIGHT = 1.0
EDGE_LOSS_WEIGHT = 1.0

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 50
SAVE_EVERY = 5

# Which datasets to use for training
DATASETS = ['DocTamper', 'CASIAv2', 'RTM', 'DEFACTO', 'MIDV500']

# =============================================================================
# SMOKE TEST MODE
# =============================================================================
# If True, trains just 2 epochs on 2 batches each, to quickly verify pipeline
SMOKE_TEST = True
SMOKE_TEST_EPOCHS = 2
SMOKE_TEST_BATCHES = 2


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize model
    model = MVSSNetLite().to(device)
    
    # 2. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Loss functions
    # Using BCEWithLogitsLoss because usually networks output raw logits 
    # instead of post-sigmoid probabilities for numerical stability.
    seg_criterion = nn.BCEWithLogitsLoss()
    edge_criterion = nn.BCEWithLogitsLoss()
    
    # 4. DataLoader
    print(f"Loading datasets: {DATASETS}...")
    train_loader = get_dataloader(DATASETS, batch_size=BATCH_SIZE, is_train=True)
    
    # Ensure output dirs exist
    os.makedirs('model/checkpoints', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    total_epochs = SMOKE_TEST_EPOCHS if SMOKE_TEST else EPOCHS
    
    # Tracking for plot
    history = {
        'epoch': [],
        'seg_loss': [],
        'edge_loss': [],
        'total_loss': []
    }
    
    print("Starting training loop...")
    for epoch in range(1, total_epochs + 1):
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
        
        print(f"=== Epoch {epoch} Summary ===")
        print(f"Avg Seg: {avg_seg:.4f} | Avg Edge: {avg_edge:.4f} | Avg Total: {avg_total:.4f}\n")
        
        history['epoch'].append(epoch)
        history['seg_loss'].append(avg_seg)
        history['edge_loss'].append(avg_edge)
        history['total_loss'].append(avg_total)
        
        # Save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == total_epochs:
            chkpt_path = f"model/checkpoints/mvss_lite_ep{epoch}.pt"
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
    plt.plot(history['epoch'], history['seg_loss'], label='Seg Loss', marker='o')
    plt.plot(history['epoch'], history['edge_loss'], label='Edge Loss', marker='o')
    plt.plot(history['epoch'], history['total_loss'], label='Total Loss', marker='o', linewidth=2)
    
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('reports/loss_curve.png')
    plt.close()
    print("Training finished! Plot saved to reports/loss_curve.png")


if __name__ == '__main__':
    train()
