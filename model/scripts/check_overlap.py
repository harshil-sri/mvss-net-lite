import torch
import cv2
import numpy as np
import os
import random

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from data_pipeline.dataset_loader import get_dataloader

def check_overlap():
    print("Extracting probe sets using get_dataloader(['RTM'])...")
    rtm_only_ds = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)[1].dataset.dataset
    rtm_val_indices = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)[1].dataset.indices
    
    with open('reports/probe_indices.txt', 'r') as f:
        lines = f.readlines()
        old_idx = [int(x) for x in lines[0].strip().replace('OLD:', '').split(',')]
        new_idx = [int(x) for x in lines[1].strip().replace('NEW:', '').split(',')]
        
    old_paths = [rtm_only_ds.samples[rtm_val_indices[i]][0] for i in old_idx]
    new_paths = [rtm_only_ds.samples[rtm_val_indices[i]][0] for i in new_idx]
    
    print("\nExtracting train set using get_dataloader(['RTM', 'MIDV500'])...")
    train_loader, val_loader, test_loader = get_dataloader(['RTM', 'MIDV500'], batch_size=1, is_train=True, return_splits=True, use_balanced_sampler=True)
    
    train_ds = train_loader.dataset.dataset
    train_indices = train_loader.dataset.indices
    
    train_paths = set([train_ds.samples[i][0] for i in train_indices])
    val_paths = set([train_ds.samples[i][0] for i in val_loader.dataset.indices])
    test_paths = set([train_ds.samples[i][0] for i in test_loader.dataset.indices])
    
    old_in_train = [p for p in old_paths if p in train_paths]
    new_in_train = [p for p in new_paths if p in train_paths]
    
    old_in_val = [p for p in old_paths if p in val_paths]
    new_in_val = [p for p in new_paths if p in val_paths]
    
    print(f"\n--- OVERLAP RESULTS ---")
    print(f"OLD Probe Set (20 images):")
    print(f"  In Train Set: {len(old_in_train)}")
    print(f"  In Val Set:   {len(old_in_val)}")
    print(f"  In Test Set:  {20 - len(old_in_train) - len(old_in_val)}")
    
    print(f"\nNEW Probe Set (20 images):")
    print(f"  In Train Set: {len(new_in_train)}")
    print(f"  In Val Set:   {len(new_in_val)}")
    print(f"  In Test Set:  {20 - len(new_in_train) - len(new_in_val)}")

if __name__ == '__main__':
    check_overlap()
