import os
import glob
import json
import random

def build_manifest():
    data_root = 'data_pipeline/raw'
    datasets_to_scan = ['CASIAv2', 'DEFACTO', 'RTM', 'MIDV500']
    
    # We will build a unified list of (img_path, mask_path, dataset_name)
    all_samples = []
    
    for dname in datasets_to_scan:
        path = os.path.join(data_root, dname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found! Skipping...")
            continue
            
        images = glob.glob(os.path.join(path, '*.jpg'))
        for img_path in sorted(images):
            mask_path = img_path.replace('.jpg', '_mask.png')
            if os.path.exists(mask_path):
                all_samples.append((img_path, mask_path, dname))
            else:
                mask_path2 = img_path.replace('.jpg', '.png')
                if os.path.exists(mask_path2):
                    all_samples.append((img_path, mask_path2, dname))
                    
    print(f"Found a total of {len(all_samples)} samples across all requested datasets.")
    
    # Shuffle with a fixed seed
    random.seed(42)
    random.shuffle(all_samples)
    
    # 80/10/10 split
    total = len(all_samples)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]
    
    manifest = {
        'train': [{'image': s[0], 'mask': s[1], 'dataset': s[2]} for s in train_samples],
        'val':   [{'image': s[0], 'mask': s[1], 'dataset': s[2]} for s in val_samples],
        'test':  [{'image': s[0], 'mask': s[1], 'dataset': s[2]} for s in test_samples]
    }
    
    os.makedirs('reports', exist_ok=True)
    manifest_path = 'reports/manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Manifest written to {manifest_path}.")
    print("\n--- Split Counts ---")
    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")
    print(f"Test:  {len(test_samples)}")
    
    print("\n--- Per-Dataset Representation ---")
    for split_name, samples in zip(['Train', 'Val', 'Test'], [train_samples, val_samples, test_samples]):
        counts = {}
        for s in samples:
            counts[s[2]] = counts.get(s[2], 0) + 1
        print(f"[{split_name}]")
        for k, v in counts.items():
            print(f"  {k}: {v}")

if __name__ == '__main__':
    build_manifest()
