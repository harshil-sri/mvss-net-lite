"""
NON-TRAINING preflight: verify the class-label contract between
data_pipeline/dataset_loader.py (producer) and model/train.py (consumer)
after the DANN label-contract patch.

No model built, no optimizer step, no checkpoint written.
"""
import sys, json, ast
sys.path.insert(0, '.')
from data_pipeline.dataset_loader import TARGET_CLASSES, compute_sample_classes

print("PRODUCER  data_pipeline/dataset_loader.py TARGET_CLASSES =", TARGET_CLASSES)

src = open('model/train.py').read()
tree = ast.parse(src)

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'log_sampler_composition':
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict) and sub.keys and all(isinstance(k, ast.Constant) for k in sub.keys):
                keys = tuple(k.value for k in sub.keys)
                print(f"CONSUMER  model/train.py:{sub.lineno} log_sampler_composition counts keys = {keys}")
                missing = [c for c in TARGET_CLASSES if c not in keys]
                print(f"          -> classes NOT counted (fall into 'other'): {missing}")

for node in ast.walk(tree):
    if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.In):
        if isinstance(node.comparators[0], ast.Tuple):
            vals = [e.value for e in node.comparators[0].elts if isinstance(e, ast.Constant)]
            if vals and any('rtm' in str(v) for v in vals):
                print(f"CONSUMER  model/train.py:{node.lineno} DANN valid_mask allowed = {tuple(vals)}")
                missing = [c for c in TARGET_CLASSES if c not in vals]
                print(f"          -> classes EXCLUDED from domain loss: {missing}")

m = json.load(open('reports/manifest.json'))
probe = []
for ds in ['RTM', 'MIDV500', 'MIDV500_synthetic']:
    hits = [s for s in m['train'] if s['dataset'] == ds][:2]
    probe += [(s['image'], s['mask']) for s in hits]
labels = compute_sample_classes(probe)
print("\nEMPIRICAL producer output on 6 real manifest samples:")
for (img, _), lab in zip(probe, labels):
    print(f"  {lab:16s} <- {img.split('/')[-1]}")

counts = dict.fromkeys(TARGET_CLASSES, 0)
other = 0
for l in labels:
    if l in counts: counts[l] += 1
    else: other += 1
print(f"\nSIMULATED log_sampler_composition -> {counts}  other={other}")

valid = [l in TARGET_CLASSES for l in labels]
targets = [1 if l.startswith('midv500') else 0 for l in labels]
kept = [t for t, v in zip(targets, valid) if v]
print(f"SIMULATED DANN valid_mask kept {sum(valid)}/{len(labels)} samples; "
      f"domain targets among kept = {kept} (distinct={set(kept)})")
if len(set(kept)) < 2:
    print("  *** DEGENERATE: domain classifier still sees only ONE class ***")
else:
    print("  OK: domain classifier sees BOTH domains (RTM & MIDV500) -> DANN is meaningful again")