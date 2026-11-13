import os
import cv2
import json
import torch
import torch.nn.functional as F
import numpy as np
import glob
import random
import re
from tqdm import tqdm
from data_pipeline.dataset_loader import get_dataloader

def bayar_noise_maps(img_tensor):
    bayar_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img_tensor.device)
    bayar_kernel = bayar_kernel / 4.0
    gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    return F.conv2d(gray, bayar_kernel, padding=1)

def laplacian_maps(img_tensor):
    lap_kernel = torch.tensor([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img_tensor.device)
    gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    return F.conv2d(gray, lap_kernel, padding=1)

def get_quad_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if 'quad' in data:
        return np.array(data['quad'], dtype=np.float32)
    return None

def compute_stats(img, mask):
    # img: [C, H, W] tensor in [0,1]
    # mask: [1, H, W] tensor 0 or 1
    # add batch dim
    img = img.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()
    
    noise = bayar_noise_maps(img)
    hf = laplacian_maps(img)
    
    m1 = mask > 0
    m0 = mask == 0
    
    if m1.sum() == 0 or m0.sum() == 0:
        return None
        
    v1 = noise[m1].var().item()
    m1_abs = noise[m1].abs().mean().item()
    h1 = hf[m1].abs().mean().item()
    
    v0 = noise[m0].var().item()
    m0_abs = noise[m0].abs().mean().item()
    h0 = hf[m0].abs().mean().item()
    
    return {
        'v1': v1, 'm1': m1_abs, 'h1': h1,
        'v0': v0, 'm0': m0_abs, 'h0': h0,
        'dv': v1 - v0, 'dm': m1_abs - m0_abs, 'dh': h1 - h0
    }

def summarize_stats(stats_list):
    res = {}
    for k in stats_list[0].keys():
        vals = [s[k] for s in stats_list]
        res[k] = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'q1': np.percentile(vals, 25),
            'q2': np.median(vals),
            'q3': np.percentile(vals, 75)
        }
    return res

def print_summary(name, summary):
    print(f"--- {name} ---")
    print("Global (Forged Region / Mask==1):")
    print(f"  Bayar Var:  {summary['v1']['mean']:.5f}")
    print(f"  Bayar Mean: {summary['m1']['mean']:.5f}")
    print(f"  Lap HF:     {summary['h1']['mean']:.5f}")
    print("Local Delta (Forged - Authentic):")
    for stat_name, k in [("Bayar Var", "dv"), ("Bayar Mean", "dm"), ("Lap HF", "dh")]:
        s = summary[k]
        print(f"  {stat_name}: Mean={s['mean']:.5f} (Std={s['std']:.5f}) | Quartiles: [{s['q1']:.5f}, {s['q2']:.5f}, {s['q3']:.5f}]")
    print()

def generate_probe_set(midv500_authentic, doc_id_to_dir, blur_setting, output_dir, num_samples=200):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    to_tamper = random.sample(midv500_authentic, num_samples * 2) # Sample more in case of skips
    
    generated = []
    op_counts = {'copy_move': 0, 'splice': 0}
    
    for img_path in to_tamper:
        base_name = os.path.basename(img_path)
        parts = base_name.split('_')
        frame_name = parts[2] + '_' + parts[3].split('.')[0]
        scene = frame_name[:2]
        doc_id = frame_name[2:4]
        
        dir_path = doc_id_to_dir.get(doc_id)
        if not dir_path: continue
        
        frame_json = os.path.join(dir_path, "ground_truth", scene, frame_name + ".json")
        template_jsons = glob.glob(os.path.join(dir_path, "ground_truth", "*.json"))
        if not template_jsons: continue
        template_json = template_jsons[0]
        
        template_imgs = glob.glob(os.path.join(dir_path, "images", "*.*"))
        if not template_imgs: continue
        template_img_path = template_imgs[0]
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        temp_img = cv2.imread(template_img_path)
        if temp_img is None: continue
        H_temp, W_temp = temp_img.shape[:2]
        template_quad = np.array([[0, 0], [W_temp, 0], [W_temp, H_temp], [0, H_temp]], dtype=np.float32)
        
        img_quad = get_quad_from_json(frame_json)
        if img_quad is None: continue
        H_t2i = cv2.getPerspectiveTransform(template_quad, img_quad)
        
        with open(template_json, 'r') as f:
            fields = json.load(f)
            
        field_keys = [k for k in fields.keys() if 'quad' in fields[k]]
        if not field_keys: continue
        
        if op_counts['copy_move'] < num_samples // 2 and op_counts['splice'] < num_samples // 2:
            op = random.choice(['copy_move', 'splice'])
        elif op_counts['copy_move'] < num_samples // 2:
            op = 'copy_move'
        else:
            op = 'splice'
            
        try:
            if op == 'copy_move':
                if len(field_keys) < 2: continue
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
                
                if blur_setting:
                    mask_blurred = cv2.GaussianBlur(mask, blur_setting[0], blur_setting[1])
                else:
                    mask_blurred = mask
                    
                mask_3d = (mask_blurred / 255.0)[..., None]
                tampered = (img * (1 - mask_3d) + warped_img * mask_3d).astype(np.uint8)
                
            else:
                donor_scene = random.choice(['CA', 'CS', 'HA', 'HS', 'KA', 'KS', 'PA', 'PS', 'TA', 'TS'])
                donor_jsons = glob.glob(os.path.join(dir_path, "ground_truth", donor_scene, "*.json"))
                if not donor_jsons: continue
                donor_json = random.choice(donor_jsons)
                donor_frame = os.path.basename(donor_json).replace('.json', '')
                donor_img_path = glob.glob(os.path.join(dir_path, "images", donor_scene, donor_frame + ".*"))
                if not donor_img_path: continue
                donor_img = cv2.imread(donor_img_path[0])
                if donor_img is None: continue
                    
                donor_quad = get_quad_from_json(donor_json)
                if donor_quad is None: continue
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
                
                if blur_setting:
                    mask_blurred = cv2.GaussianBlur(mask, blur_setting[0], blur_setting[1])
                else:
                    mask_blurred = mask
                    
                mask_3d = (mask_blurred / 255.0)[..., None]
                tampered = (img * (1 - mask_3d) + warped_img * mask_3d).astype(np.uint8)
        except Exception as e:
            continue
            
        op_counts[op] += 1
        
        idx = len(generated)
        out_img = os.path.join(output_dir, f"{idx:04d}.jpg")
        out_msk = os.path.join(output_dir, f"{idx:04d}_mask.png")
        cv2.imwrite(out_img, tampered)
        cv2.imwrite(out_msk, mask) # Save crisp mask
        generated.append((out_img, out_msk, img_path))
        
        if len(generated) >= num_samples:
            break
            
    return generated

def run_probe():
    import torchvision.transforms as T
    to_tensor = T.ToTensor()
    
    with open('reports/manifest.json', 'r') as f:
        manifest = json.load(f)
        
    midv500_auth = []
    for split in ['train', 'val', 'test']:
        for s in manifest[split]:
            if s['dataset'] == 'MIDV500':
                midv500_auth.append(s['image'])
    midv500_auth = [f for f in midv500_auth if re.match(r'.*midv500_[A-Z]{2}_[A-Z]{2}[0-9]{2}_[0-9]{2}\.jpg', f)]
    
    doc_id_to_dir = {}
    for d in glob.glob("data/MIDV-500/midv500/*"):
        if os.path.isdir(d):
            doc_id = os.path.basename(d).split('_')[0]
            if doc_id.isdigit():
                doc_id_to_dir[doc_id] = d
                
    blur_settings = {
        'A (Current Default)': ((5, 5), 1.5),
        'B (No Blur)': None,
        'C (Intermediate)': ((3, 3), 0.75)
    }
    
    # Pre-compute RTM stats (real forged)
    print("Computing RTM Real Forged Stats...")
    rtm_loader = get_dataloader('RTM', batch_size=1, is_train=False)
    rtm_stats = []
    for img, mask, _ in tqdm(rtm_loader, total=200):
        if mask.sum() == 0: continue
        st = compute_stats(img[0], mask[0])
        if st: rtm_stats.append(st)
        if len(rtm_stats) >= 200: break
    rtm_summary = summarize_stats(rtm_stats)
    
    for name, bset in blur_settings.items():
        print(f"\n=========================================")
        print(f"Testing Blur Setting: {name} {bset}")
        print(f"=========================================")
        
        out_dir = f"data_pipeline/probe_blur_{name.split(' ')[0]}"
        gen_list = generate_probe_set(midv500_auth, doc_id_to_dir, bset, out_dir, num_samples=200)
        
        synth_stats = []
        auth_stats = []
        
        # Load and compute stats
        for tampered_img_path, mask_path, orig_img_path in gen_list:
            t_img = cv2.imread(tampered_img_path)
            o_img = cv2.imread(orig_img_path)
            m_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # The dataloader resizes to 256x256 before tensor conversion! We must do exactly that.
            t_img = cv2.resize(t_img, (256, 256))
            o_img = cv2.resize(o_img, (256, 256))
            # Wait, dataloader resizes mask using F.adaptive_max_pool2d. We can use cv2.resize with max interpolation (or nearest)
            m_img = cv2.resize(m_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            t_img_t = to_tensor(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
            o_img_t = to_tensor(cv2.cvtColor(o_img, cv2.COLOR_BGR2RGB))
            m_img_t = to_tensor(m_img) # [1, H, W]
            
            st_t = compute_stats(t_img_t, m_img_t)
            st_o = compute_stats(o_img_t, m_img_t)
            
            if st_t: synth_stats.append(st_t)
            if st_o: auth_stats.append(st_o)
            
        print_summary("Real RTM-Forged (Reference)", rtm_summary)
        print_summary(f"Synthetic MIDV500-Forged", summarize_stats(synth_stats))
        print_summary("MIDV500-Authentic Baseline (Same regions)", summarize_stats(auth_stats))

if __name__ == '__main__':
    run_probe()
