import torch
import matplotlib.pyplot as plt
import os
from data_pipeline.dataset_loader import get_dataloader

def test_dataloader():
    print("Initializing DataLoader Smoke Test...")
    
    # We will test all datasets you plan to use
    datasets_to_test = ['CASIAv2', 'RTM', 'DEFACTO', 'MIDV500']
    
    try:
        # Create the dataloader with a small batch size
        loader = get_dataloader(datasets_to_test, batch_size=4, is_train=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize dataloader. Make sure the folders exist in data_pipeline/raw/!\\nDetails: {e}")
        return

    print(f"DataLoader initialized successfully. Total batches available: {len(loader)}")
    
    if len(loader) == 0:
        print("WARNING: DataLoader is empty! Did your formatting scripts output to the correct folders in data_pipeline/raw/?")
        return

    print("Fetching 5 sample batches to verify stability...")
    
    try:
        for i, (imgs, masks, edges) in enumerate(loader):
            if i >= 5:
                break
            
            print(f"\\n--- Batch {i+1} ---")
            print(f"Images Shape: {imgs.shape}, Min: {imgs.min().item():.2f}, Max: {imgs.max().item():.2f}")
            print(f"Masks Shape:  {masks.shape}, Unique Values: {torch.unique(masks).tolist()}")
            print(f"Edges Shape:  {edges.shape}, Unique Values: {torch.unique(edges).tolist()}")
            
            # For the very first batch, let's plot it and save it to visually verify
            if i == 0:
                print("Saving a visualization of the first batch to reports/smoke_test_batch.png...")
                batch_size = imgs.size(0)
                fig, axes = plt.subplots(batch_size, 3, figsize=(10, 3 * batch_size))
                
                # Handle case where batch_size is 1
                if batch_size == 1:
                    axes = [axes]
                    
                for b in range(batch_size):
                    # Convert tensors back to plottable format
                    img_plot = imgs[b].permute(1, 2, 0).numpy()
                    mask_plot = masks[b].squeeze().numpy()
                    edge_plot = edges[b].squeeze().numpy()
                    
                    axes[b][0].imshow(img_plot)
                    axes[b][0].set_title(f"Image {b+1}")
                    axes[b][0].axis('off')
                    
                    axes[b][1].imshow(mask_plot, cmap='gray')
                    axes[b][1].set_title(f"Mask {b+1}")
                    axes[b][1].axis('off')
                    
                    axes[b][2].imshow(edge_plot, cmap='gray')
                    axes[b][2].set_title(f"Edge {b+1}")
                    axes[b][2].axis('off')
                
                plt.tight_layout()
                os.makedirs('reports', exist_ok=True)
                plt.savefig('reports/smoke_test_batch.png')
                plt.close()
                
    except Exception as e:
        print(f"\\nCRITICAL ERROR during batch fetching! The pipeline crashed:\\n{e}")
        return

    print("\\nSUCCESS! The data pipeline successfully loaded, augmented, and formatted batches without crashing. You are ready to train.")

if __name__ == "__main__":
    test_dataloader()
