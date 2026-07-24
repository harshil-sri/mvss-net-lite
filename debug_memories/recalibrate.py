import torch
import cv2
import numpy as np
import os
import glob
from collections import Counter

torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)

from data_pipeline.dataset_loader import get_dataloader

def verify_sampler_and_compute_weight():
    print("--- 1. Verifying Sampler & Computing Effective pos_weight ---")
    train_loader, _, _ = get_dataloader(['RTM', 'MIDV500'], batch_size=8, is_train=True, return_splits=True, use_balanced_sampler=True)
    
    # We will simulate 50 batches (400 images)
    num_batches = 50
    rtm_auth_count = 0
    rtm_forg_count = 0
    midv_auth_count = 0
    
    total_pos_pixels = 0
    total_neg_pixels = 0
    total_seg_pos_pixels = 0
    total_seg_neg_pixels = 0
    
    for i, (imgs, masks, edges) in enumerate(train_loader):
        if i >= num_batches:
            break
            
        pos_pixels = edges.sum().item()
        neg_pixels = edges.numel() - pos_pixels
        total_pos_pixels += pos_pixels
        total_neg_pixels += neg_pixels
        
        seg_pos_pixels = masks.sum().item()
        seg_neg_pixels = masks.numel() - seg_pos_pixels
        total_seg_pos_pixels += seg_pos_pixels
        total_seg_neg_pixels += seg_neg_pixels
        
    print(f"Scanned {num_batches} batches.")
    print(f"Total Positive Edge Pixels: {total_pos_pixels}")
    print(f"Total Negative Edge Pixels: {total_neg_pixels}")
    if total_pos_pixels > 0:
        effective_weight = total_neg_pixels / total_pos_pixels
        print(f"EFFECTIVE EDGE pos_weight under new sampler: {effective_weight:.2f}")
    
    print(f"Total Positive Seg Pixels: {total_seg_pos_pixels}")
    print(f"Total Negative Seg Pixels: {total_seg_neg_pixels}")
    if total_seg_pos_pixels > 0:
        seg_effective_weight = total_seg_neg_pixels / total_seg_pos_pixels
        print(f"EFFECTIVE SEG pos_weight under new sampler: {seg_effective_weight:.2f}")
    else:
        print("No positive pixels found!")

def get_dataloader_with_paths():
    # A temporary hack to get paths to verify the sampler composition
    from data_pipeline.dataset_loader import ForgeryDataset
    from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
    
    ds = ForgeryDataset('data_pipeline/raw', ['RTM', 'MIDV500'], is_train=True)
    total = len(ds)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_ds, _, _ = random_split(ds, [train_size, val_size, test_size])
    
    weights = []
    for idx in train_ds.indices:
        img_path, mask_path = ds.samples[idx]
        is_forged = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).sum() > 0
        is_rtm = 'RTM' in img_path
        is_midv = 'MIDV500' in img_path
        
        if is_rtm and is_forged:
            weights.append(1.0 / 6000)
        elif is_rtm and not is_forged:
            weights.append(1.0 / 3000)
        elif is_midv:
            weights.append(1.0 / 15050)
        else:
            weights.append(1.0)
            
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    # To get paths, we iterate over the sampler indices directly
    return ds, train_ds, sampler

def test_sampler_composition():
    print("\n--- 2. Verifying Sampler Composition ---")
    ds, train_ds, sampler = get_dataloader_with_paths()
    
    # Draw 400 samples (50 batches of 8)
    drawn_indices = list(sampler)[:400]
    
    rtm_forg = 0
    rtm_auth = 0
    midv_auth = 0
    
    for idx in drawn_indices:
        # sampler yields an index into train_ds
        # train_ds is a Subset, so we need train_ds.indices[idx] to get the absolute ds index
        abs_idx = train_ds.indices[idx]
        img_path, mask_path = ds.samples[abs_idx]
        is_forged = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).sum() > 0
        is_rtm = 'RTM' in img_path
        is_midv = 'MIDV500' in img_path
        
        if is_rtm and is_forged:
            rtm_forg += 1
        elif is_rtm and not is_forged:
            rtm_auth += 1
        elif is_midv:
            midv_auth += 1
            
    total = len(drawn_indices)
    print(f"Out of {total} sampled images:")
    print(f"RTM Forged: {rtm_forg} ({rtm_forg/total*100:.1f}%)")
    print(f"RTM Authentic: {rtm_auth} ({rtm_auth/total*100:.1f}%)")
    print(f"MIDV500 Authentic: {midv_auth} ({midv_auth/total*100:.1f}%)")

if __name__ == '__main__':
    verify_sampler_and_compute_weight()
    test_sampler_composition()
