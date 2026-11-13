import re

with open('diagnostics/probe_blur_v3.py', 'r') as f:
    code = f.read()

imports = """
import traceback
import psutil
import time
"""
code = code.replace("import os\n", "import os\n" + imports)

# We want to replace the `run_probe()` loops.

# Replace run_probe completely to ensure it's correct.
old_run_probe = code[code.find('def run_probe():'):]

new_run_probe = """def run_probe():
    import torchvision.transforms as T
    to_tensor = T.ToTensor()
    
    with open('reports/manifest.json', 'r') as f:
        manifest = json.load(f)
        
    midv500_auth = []
    for split in ['train', 'val', 'test']:
        for s in manifest[split]:
            if s['dataset'] == 'MIDV500':
                midv500_auth.append(s['image'])
    midv500_auth = [f for f in midv500_auth if re.match(r'.*midv500_[A-Z]{2}_[A-Z]{2}[0-9]{2}_[0-9]{2}\\.jpg', f)]
    
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
    
    print("Computing RTM Real Forged Stats...")
    rtm_loader = get_dataloader('RTM', batch_size=1, is_train=False)
    rtm_stats = []
    for img, mask, _ in tqdm(rtm_loader, total=200):
        if mask.sum() == 0: continue
        st = compute_stats(img[0], mask[0])
        if st: rtm_stats.append(st)
        if len(rtm_stats) >= 200: break
    rtm_summary = summarize_stats(rtm_stats)
    
    log_dir = 'diagnostics/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'probe_blur_v3_run.log')
    
    # Initialize the log file
    with open(log_file, 'a') as lf:
        lf.write(f"\\n--- Starting new run at {time.time()} ---\\n")
        
    for name, bset in blur_settings.items():
        print(f"\\n=========================================")
        print(f"Testing Blur Setting: {name} {bset}")
        print(f"=========================================")
        
        # SMOKETEST: new dir, num_samples=7
        out_dir = f"data_pipeline/probe_blur_v3_smoketest_{name.split(' ')[0]}"
        gen_list = generate_probe_set(midv500_auth, doc_id_to_dir, bset, out_dir, num_samples=7)
        
        synth_stats = []
        auth_stats = []

        tampered_samples = [(t, m) for (t, m, o) in gen_list]
        authentic_samples = [(o, m) for (t, m, o) in gen_list]

        ds_tampered = ForgeryDataset(tampered_samples, crop_size=256, is_train=False)
        ds_authentic = ForgeryDataset(authentic_samples, crop_size=256, is_train=False)

        start_time = time.time()
        
        for i in range(len(gen_list)):
            try:
                t_img_t, m_img_t, _ = ds_tampered[i]
                o_img_t, m_img_o, _ = ds_authentic[i]

                st_t = compute_stats(t_img_t, m_img_t)
                st_o = compute_stats(o_img_t, m_img_o)

                if st_t: synth_stats.append(st_t)
                if st_o: auth_stats.append(st_o)
                
                # Incremental write
                with open(os.path.join(log_dir, 'incremental_stats.jsonl'), 'a') as f_inc:
                    f_inc.write(json.dumps({'setting': name, 'index': i, 'st_t': st_t, 'st_o': st_o}) + '\\n')
                
                elapsed = time.time() - start_time
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                
                with open(log_file, 'a') as lf:
                    lf.write(f"Setting: {name} | Image: {i}/{len(gen_list)} | Elapsed: {elapsed:.2f}s | RSS: {mem:.2f} MB\\n")
                    
            except Exception as e:
                with open(log_file, 'a') as lf:
                    lf.write(f"\\nEXCEPTION in Setting: {name} | Image: {i}\\n")
                    lf.write(traceback.format_exc())
                    lf.write("\\n")
                raise

        if synth_stats:
            print_summary("Real RTM-Forged (Reference)", rtm_summary)
            print_summary(f"Synthetic MIDV500-Forged", summarize_stats(synth_stats))
            print_summary("MIDV500-Authentic Baseline (Same regions)", summarize_stats(auth_stats))

if __name__ == '__main__':
    run_probe()
"""

code = code.replace(old_run_probe, new_run_probe)

with open('diagnostics/probe_blur_v3.py', 'w') as f:
    f.write(code)
