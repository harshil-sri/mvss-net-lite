import os
import cv2
import json
import numpy as np
import glob
import random
import re
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def get_quad_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if 'quad' in data:
        return np.array(data['quad'], dtype=np.float32)
    return None

def process_sample(args):
    idx, img_path, doc_id_to_dir, out_img_dir, out_msk_dir = args
    base_name = os.path.basename(img_path)
    parts = base_name.split('_')
    frame_name = parts[2] + '_' + parts[3].split('.')[0]
    scene = frame_name[:2]
    doc_id = frame_name[2:4]
    
    dir_path = doc_id_to_dir[doc_id]
    frame_json = os.path.join(dir_path, "ground_truth", scene, frame_name + ".json")
    template_jsons = glob.glob(os.path.join(dir_path, "ground_truth", "*.json"))
    if not template_jsons: return None
    template_json = template_jsons[0]
    
    template_imgs = glob.glob(os.path.join(dir_path, "images", "*.*"))
    if not template_imgs: return None
    template_img_path = template_imgs[0]
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    
    temp_img = cv2.imread(template_img_path)
    if temp_img is None: return None
    H_temp, W_temp = temp_img.shape[:2]
    template_quad = np.array([[0, 0], [W_temp, 0], [W_temp, H_temp], [0, H_temp]], dtype=np.float32)
    
    img_quad = get_quad_from_json(frame_json)
    if img_quad is None: return None
    H_t2i = cv2.getPerspectiveTransform(template_quad, img_quad)
    
    with open(template_json, 'r') as f:
        fields = json.load(f)
        
    field_keys = [k for k in fields.keys() if 'quad' in fields[k]]
    if not field_keys: return None
    
    op = random.choice(['copy_move', 'splice'])
    
    if op == 'copy_move':
        if len(field_keys) < 2:
            return None
        src_k, dst_k = random.sample(field_keys, 2)
        src_t_quad = np.array(fields[src_k]['quad'], dtype=np.float32)
        dst_t_quad = np.array(fields[dst_k]['quad'], dtype=np.float32)
        
        src_i_quad = cv2.perspectiveTransform(src_t_quad.reshape(-1, 1, 2), H_t2i).reshape(4, 2)
        dst_i_quad = cv2.perspectiveTransform(dst_t_quad.reshape(-1, 1, 2), H_t2i).reshape(4, 2)
        
        H_s2d = cv2.getPerspectiveTransform(src_i_quad, dst_i_quad)
        warped_img = cv2.warpPerspective(img, H_s2d, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        brightness_shift = np.random.randint(-15, 15)
        warped_img = np.clip(warped_img.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [dst_i_quad.astype(np.int32)], 255)
        
        mask_blurred = cv2.GaussianBlur(mask, (5, 5), 1.5)
        mask_3d = (mask_blurred / 255.0)[..., None]
        
        tampered = (img * (1 - mask_3d) + warped_img * mask_3d).astype(np.uint8)
        
    else:
        donor_scene = random.choice(['CA', 'CS', 'HA', 'HS', 'KA', 'KS', 'PA', 'PS', 'TA', 'TS'])
        donor_jsons = glob.glob(os.path.join(dir_path, "ground_truth", donor_scene, "*.json"))
        if not donor_jsons: return None
        donor_json = random.choice(donor_jsons)
        donor_frame = os.path.basename(donor_json).replace('.json', '')
        donor_img_path = glob.glob(os.path.join(dir_path, "images", donor_scene, donor_frame + ".*"))
        if not donor_img_path: return None
        donor_img = cv2.imread(donor_img_path[0])
        if donor_img is None: return None
            
        donor_quad = get_quad_from_json(donor_json)
        if donor_quad is None: return None
        H_t2d = cv2.getPerspectiveTransform(template_quad, donor_quad)
        
        target_k = random.choice(field_keys)
        field_t_quad = np.array(fields[target_k]['quad'], dtype=np.float32)
        
        donor_field_quad = cv2.perspectiveTransform(field_t_quad.reshape(-1, 1, 2), H_t2d).reshape(4, 2)
        target_field_quad = cv2.perspectiveTransform(field_t_quad.reshape(-1, 1, 2), H_t2i).reshape(4, 2)
        
        H_d2t = cv2.getPerspectiveTransform(donor_field_quad, target_field_quad)
        warped_img = cv2.warpPerspective(donor_img, H_d2t, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        brightness_shift = np.random.randint(-15, 15)
        warped_img = np.clip(warped_img.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [target_field_quad.astype(np.int32)], 255)
        
        mask_blurred = cv2.GaussianBlur(mask, (5, 5), 1.5)
        mask_3d = (mask_blurred / 255.0)[..., None]
        
        tampered = (img * (1 - mask_3d) + warped_img * mask_3d).astype(np.uint8)

    new_name = base_name.replace('.jpg', f'_synthetic_{idx:04d}.jpg')
    out_img = os.path.join(out_img_dir, new_name)
    out_msk = os.path.join(out_msk_dir, new_name.replace('.jpg', '_mask.png'))
    
    cv2.imwrite(out_img, tampered)
    cv2.imwrite(out_msk, mask)
    
    return {
        'image': out_img,
        'mask': out_msk,
        'dataset': 'MIDV500_synthetic',
        'op': op,
        'original_image': img_path
    }

def main():
    manifest_path = 'reports/manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    midv500_authentic = []
    for split in ['train', 'val', 'test']:
        for s in manifest[split]:
            if s['dataset'] == 'MIDV500':
                midv500_authentic.append(s['image'])
                
    midv500_authentic = [f for f in midv500_authentic if re.match(r'.*midv500_[A-Z]{2}_[A-Z]{2}[0-9]{2}_[0-9]{2}\.jpg', f)]
                
    random.seed(42)
    np.random.seed(42)
    
    to_tamper = random.sample(midv500_authentic, 5000)
    
    doc_id_to_dir = {}
    for d in glob.glob("data/MIDV-500/midv500/*"):
        if os.path.isdir(d):
            base = os.path.basename(d)
            doc_id = base.split('_')[0]
            if doc_id.isdigit():
                doc_id_to_dir[doc_id] = d
                
    out_img_dir = 'data_pipeline/raw/MIDV500_synthetic/images'
    out_msk_dir = 'data_pipeline/raw/MIDV500_synthetic/masks'
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)
    
    synthetic_samples = []
    
    args_list = [(idx, img, doc_id_to_dir, out_img_dir, out_msk_dir) for idx, img in enumerate(to_tamper)]
    
    print("Generating forgeries with multiprocessing...")
    with Pool(processes=cpu_count()) as p:
        results = list(tqdm(p.imap(process_sample, args_list), total=len(args_list)))
        
    synthetic_samples = [r for r in results if r is not None]

    print(f"\nGenerated {len(synthetic_samples)} synthetic samples.")
    with open('reports/synthetic_meta.json', 'w') as f:
        json.dump(synthetic_samples, f)

if __name__ == '__main__':
    main()
