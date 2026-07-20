import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ForgeryDataset(Dataset):
    # Just a simple dataset for all our forgery data
    def __init__(self, data_root, dataset_names, crop_size=256, is_train=True):
        self.data_root = data_root
        self.crop_size = crop_size
        self.is_train = is_train
        self.samples = []
        
        # loop over the requested datasets and find all images and masks
        for dname in dataset_names:
            path = os.path.join(data_root, dname)
            if not os.path.exists(path):
                print(f"Warning: {path} not found!")
                continue
                
            # assuming all images are jpg and masks are png
            images = glob.glob(os.path.join(path, '*.jpg'))
            for img_path in images:
                # masks usually have _mask appended before extension
                mask_path = img_path.replace('.jpg', '_mask.png')
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
                else:
                    # try without _mask just in case
                    mask_path2 = img_path.replace('.jpg', '.png')
                    if os.path.exists(mask_path2):
                        self.samples.append((img_path, mask_path2))
                    
        print(f"Loaded {len(self.samples)} samples from {dataset_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # resize mask to image size if they don't match for some reason
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        h, w = img.shape[:2]
        
        if self.is_train:
            # Random crop
            if h > self.crop_size and w > self.crop_size:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
                img = img[y:y+self.crop_size, x:x+self.crop_size]
                mask = mask[y:y+self.crop_size, x:x+self.crop_size]
            else:
                img = cv2.resize(img, (self.crop_size, self.crop_size))
                mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
                
            # Random horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
                
            # simulate jpeg compression artifacts since models overfit to them easily
            if random.random() > 0.5:
                q = random.randint(50, 95)
                # cv2 uses BGR for encode so we gotta convert back and forth
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                dec = cv2.imdecode(enc, 1)
                img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        else:
            # just resize for test
            img = cv2.resize(img, (self.crop_size, self.crop_size))
            mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
            
        # make mask binary
        mask = (mask > 127).astype(np.uint8)
        
        # get edge map via simple morphological gradient (similar to canny but easier on binary)
        kernel = np.ones((3,3), np.uint8)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        
        # convert to torch tensors [C, H, W]
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        edge_tensor = torch.from_numpy(edge).float().unsqueeze(0)
        
        return img_tensor, mask_tensor, edge_tensor

def get_dataloader(dataset_names, batch_size=4, is_train=True):
    # wrapper to get the dataloader easily
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
        
    ds = ForgeryDataset('data_pipeline/raw', dataset_names, is_train=is_train)
    return DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=0)

if __name__ == '__main__':
    # test script to make sure it works and generate progress update
    datasets = ['CASIAv2', 'DocTamper', 'Defacto', 'MIDV-500']
    
    # load a small batch of 4
    loader = get_dataloader(datasets, batch_size=4)
    images, masks, edges = next(iter(loader))
    
    os.makedirs('reports', exist_ok=True)
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i in range(images.shape[0]):
        # tensor to numpy
        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].squeeze(0).numpy()
        edge = edges[i].squeeze(0).numpy()
        
        # make overlay (red tint on mask)
        overlay = img.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Sample {i} Image')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'Mask Overlay')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(edge, cmap='gray')
        axes[2, i].set_title(f'Edge Map')
        axes[2, i].axis('off')
        
    plt.tight_layout()
    plt.savefig('reports/dataset_samples.png')
    print("Saved progress image to reports/dataset_samples.png")
