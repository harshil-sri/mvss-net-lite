import re
import csv
import json
import random
from collections import defaultdict

def extract_log_data():
    with open('reports/tmux_stage2.log', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    pos_weight = ""
    manifest_loaded = False
    num_workers_flag = ""
    errors = []
    sampler_lines = []

    for i, line in enumerate(lines):
        if "pos_weight" in line.lower():
            pos_weight += line
        if "manifest.json" in line:
            manifest_loaded = True
        if "num_workers" in line:
            num_workers_flag += line
        if "sampler" in line.lower() and "batch" in line.lower() and "rtm" in line.lower():
            sampler_lines.append(line)
        if any(err in line.lower() for err in ["error", "warning", "oom", "out of memory", "exception", "traceback"]):
            # exclude some standard warnings if they are verbose, but we'll collect all
            errors.append((i, line))

    print("=== POS WEIGHT ===")
    print(pos_weight)
    print("\n=== MANIFEST LOADED ===")
    print(manifest_loaded)
    print("\n=== NUM WORKERS ===")
    print(num_workers_flag)
    print("\n=== SAMPLER LINES (Sample of 3) ===")
    if len(sampler_lines) >= 3:
        print(sampler_lines[0].strip())
        print(sampler_lines[len(sampler_lines)//2].strip())
        print(sampler_lines[-1].strip())
    else:
        for s in sampler_lines: print(s.strip())

    print("\n=== ERRORS ===")
    for idx, err in errors[:20]:
        print(f"Line {idx}: {err.strip()}")
    if len(errors) > 20:
        print(f"... and {len(errors) - 20} more")

if __name__ == '__main__':
    extract_log_data()
