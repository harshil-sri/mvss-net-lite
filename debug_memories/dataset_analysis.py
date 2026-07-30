import os
import glob
import cv2
import numpy as np
import concurrent.futures

def check_mask(mask_path):
    if not os.path.exists(mask_path):
        return 'missing'
    # Use cv2.IMREAD_UNCHANGED for speed, assume grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 'missing'
    # For authentic images, mask should be entirely 0
    if cv2.countNonZero(mask) == 0:
        return 'authentic'
    return 'forged'

def count_dataset(dataset_name):
    path = os.path.join('data_pipeline/raw', dataset_name)
    if not os.path.exists(path):
        return dataset_name, 0, 0, 0
    
    images = glob.glob(os.path.join(path, '*.jpg'))
    if not images:
        return dataset_name, 0, 0, 0

    mask_paths = []
    for img_path in images:
        m1 = img_path.replace('.jpg', '_mask.png')
        m2 = img_path.replace('.jpg', '.png')
        if os.path.exists(m1):
            mask_paths.append(m1)
        elif os.path.exists(m2):
            mask_paths.append(m2)
        else:
            mask_paths.append('none')

    authentic = 0
    forged = 0
    missing = 0
    
    # Process in parallel for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(check_mask, mask_paths)
        for r in results:
            if r == 'authentic':
                authentic += 1
            elif r == 'forged':
                forged += 1
            else:
                missing += 1
                
    return dataset_name, len(images), authentic, forged, missing

if __name__ == "__main__":
    datasets = ['RTM', 'MIDV500', 'CASIAv2', 'DEFACTO']
    print("Analyzing dataset composition (Authentic vs Forged)...")
    for ds in datasets:
        name, total, auth, forg, miss = count_dataset(ds)
        print(f"[{name}] Total: {total} | Authentic: {auth} | Forged: {forg} | Missing Masks: {miss}")
