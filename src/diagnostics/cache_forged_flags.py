"""One-time cache of forged/authentic flags for pilot domains (mask reads are slow).
Writes reports/forged_flags.json : {split: {image_path: 0/1 forged}} for RTM+CASIAv2.
"""
import json, cv2, os
DOMAINS = ['RTM', 'CASIAv2']
m = json.load(open('reports/manifest.json'))
out = {}
for split in ['train', 'test']:
    d = {}
    n = 0
    for s in m[split]:
        if s['dataset'] not in DOMAINS:
            continue
        msk = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
        d[s['image']] = int(msk is not None and msk.sum() > 0)
        n += 1
        if n % 2000 == 0:
            print(f"{split}: {n} done")
    out[split] = d
    print(f"{split}: {n} total, forged={sum(d.values())}")
os.makedirs('reports', exist_ok=True)
json.dump(out, open('reports/forged_flags.json', 'w'))
print("saved reports/forged_flags.json")
