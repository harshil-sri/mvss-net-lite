import re

with open('diagnostics/probe_blur_v3.py', 'r') as f:
    code = f.read()

# Add imports
imports = "import traceback\nimport psutil\nimport time\n"
code = code.replace("import os\n", "import os\n" + imports)

# -----------------
# modify generate_probe_set
# -----------------
# 1. Change signature to initialize skip_count
code = code.replace(
    "    op_counts = {'copy_move': 0, 'splice': 0}\n",
    "    op_counts = {'copy_move': 0, 'splice': 0}\n    skip_count = 0\n    log_file = 'diagnostics/logs/probe_blur_v3_run.log'\n"
)

# 2. Modify loop to get index
code = code.replace(
    "    for img_path in to_tamper:\n",
    "    for idx_in_list, img_path in enumerate(to_tamper):\n"
)

# 3. Modify except Exception as e:
original_except = """        except Exception as e:
            continue"""

new_except = """        except Exception as e:
            skip_count += 1
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a') as lf:
                lf.write(f"\\nEXCEPTION in generate_probe_set | Image Index: {idx_in_list} | Path: {img_path}\\n")
                lf.write(traceback.format_exc())
                lf.write("\\n")
            continue"""
code = code.replace(original_except, new_except)

# 4. Add deliberate crash for smoketest at the START of the try block
original_try = "        try:\n"
new_try = """        try:
            # DELIBERATE SMOKE TEST SKIP
            if idx_in_list == 0:
                raise RuntimeError("Deliberate smoke test skip triggered")
"""
code = code.replace(original_try, new_try)

# 5. Return skip_count
code = code.replace(
    "    return generated\n",
    "    return generated, skip_count\n"
)

# -----------------
# modify run_probe
# -----------------
# 1. Signature
code = code.replace(
    "def run_probe():\n",
    "def run_probe(num_samples=200, smoketest=False):\n"
)

# 2. Log initialization before RTM stats
original_rtm = "    # Pre-compute RTM stats (real forged)\n    print(\"Computing RTM Real Forged Stats...\")"
new_rtm = """    log_dir = 'diagnostics/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'probe_blur_v3_run.log')
    with open(log_file, 'a') as lf:
        lf.write(f"\\n--- Starting new run at {time.time()} (Smoketest: {smoketest}) ---\\n")
        
    # Pre-compute RTM stats (real forged)
    print("Computing RTM Real Forged Stats...")"""
code = code.replace(original_rtm, new_rtm)

# 3. Instrument RTM loop
original_rtm_loop = """    for img, mask, _ in tqdm(rtm_loader, total=200):
        if mask.sum() == 0: continue
        st = compute_stats(img[0], mask[0])
        if st: rtm_stats.append(st)
        if len(rtm_stats) >= 200: break"""
new_rtm_loop = """    for img, mask, _ in tqdm(rtm_loader, total=num_samples):
        try:
            if mask.sum() == 0: continue
            st = compute_stats(img[0], mask[0])
            if st: rtm_stats.append(st)
            if len(rtm_stats) >= num_samples: break
        except Exception as e:
            with open(log_file, 'a') as lf:
                lf.write(f"\\nEXCEPTION in RTM Stats Loop\\n")
                lf.write(traceback.format_exc())
                lf.write("\\n")
            raise"""
code = code.replace(original_rtm_loop, new_rtm_loop)

# 4. Modify settings loop start
original_settings_start = """        out_dir = f"data_pipeline/probe_blur_v3_{name.split(' ')[0]}"
        gen_list = generate_probe_set(midv500_auth, doc_id_to_dir, bset, out_dir, num_samples=200)"""

new_settings_start = """        if smoketest:
            out_dir = f"data_pipeline/probe_blur_v3_smoketest_{name.split(' ')[0]}"
        else:
            out_dir = f"data_pipeline/probe_blur_v3_{name.split(' ')[0]}"
            
        gen_list, skip_count = generate_probe_set(midv500_auth, doc_id_to_dir, bset, out_dir, num_samples=num_samples)"""
code = code.replace(original_settings_start, new_settings_start)

# 5. Modify per-image inner loop
original_inner_loop = """        for i in range(len(gen_list)):
            t_img_t, m_img_t, _ = ds_tampered[i]
            o_img_t, m_img_o, _ = ds_authentic[i]

            st_t = compute_stats(t_img_t, m_img_t)
            st_o = compute_stats(o_img_t, m_img_o)

            if st_t: synth_stats.append(st_t)
            if st_o: auth_stats.append(st_o)"""

new_inner_loop = """        start_time = time.time()
        for i in range(len(gen_list)):
            try:
                t_img_t, m_img_t, _ = ds_tampered[i]
                o_img_t, m_img_o, _ = ds_authentic[i]

                st_t = compute_stats(t_img_t, m_img_t)
                st_o = compute_stats(o_img_t, m_img_o)

                if st_t: synth_stats.append(st_t)
                if st_o: auth_stats.append(st_o)
                
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
                raise"""
code = code.replace(original_inner_loop, new_inner_loop)

# 6. Summary reporting
original_summary = """        print_summary("Real RTM-Forged (Reference)", rtm_summary)
        print_summary(f"Synthetic MIDV500-Forged", summarize_stats(synth_stats))
        print_summary("MIDV500-Authentic Baseline (Same regions)", summarize_stats(auth_stats))"""
new_summary = """        print(f"--- Setting Summary: {name} ---")
        print(f"  Target Samples: {num_samples}")
        print(f"  Successfully Generated: {len(gen_list)}")
        print(f"  Failed/Skipped: {skip_count}")
        print_summary("Real RTM-Forged (Reference)", rtm_summary)
        print_summary(f"Synthetic MIDV500-Forged", summarize_stats(synth_stats))
        print_summary("MIDV500-Authentic Baseline (Same regions)", summarize_stats(auth_stats))"""
code = code.replace(original_summary, new_summary)

# 7. __main__ execution
code = code.replace(
    "if __name__ == '__main__':\n    run_probe()\n",
    "if __name__ == '__main__':\n    run_probe(num_samples=7, smoketest=True)\n"
)

with open('diagnostics/probe_blur_v3.py', 'w') as f:
    f.write(code)

