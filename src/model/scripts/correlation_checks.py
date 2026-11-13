import cv2
import numpy as np
import os
import glob

def check_forgery_location():
    print("--- 2. Checking fraction of forgeries located ON dense table structures ---")
    data_root = 'data_pipeline/raw/RTM'
    images = glob.glob(os.path.join(data_root, '*.jpg'))
    
    forged_images = []
    for img_path in images:
        mask_path = img_path.replace('.jpg', '_mask.png')
        if not os.path.exists(mask_path):
            mask_path = img_path.replace('.jpg', '.png')
        if not os.path.exists(mask_path):
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.sum() > 0:
            forged_images.append((img_path, mask))
        
        if len(forged_images) == 200:
            break
            
    print(f"Found {len(forged_images)} forged images. Processing...")
    
    table_forgeries = 0
    total = 0
    for img_path, mask in forged_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 100, 200)
        
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        
        roi_edges = edges[y1:y2, x1:x2]
        roi_density = np.sum(roi_edges > 0) / roi_edges.size if roi_edges.size > 0 else 0
        
        if roi_density > 0.04:
            table_forgeries += 1
        total += 1
        
    print(f"Forgeries located ON dense tabular/text structures: {table_forgeries}/{total} ({table_forgeries/total*100:.1f}%)")

if __name__ == '__main__':
    check_forgery_location()
