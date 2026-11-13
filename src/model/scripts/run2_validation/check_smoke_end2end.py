"""
Check 6b: END-TO-END entrypoint proof via the REAL train.py CLI.

Runs exactly 2 epochs x 2 batches (--smoke-test) on the RTM+MIDV500 manifest
split with the full Run 2 configuration (balanced sampler + index tracking +
layer freezing + init from stage1 ep45). This is a wiring proof, NOT a
training run: 4 optimizer steps total, then cleanup of all generated artifacts
(checkpoints/CSV/plot are deleted after their contents are verified).

Proves:
  - pos_weight live rescan fires at startup and prints scanned values
  - freeze report prints per-branch trainable/frozen counts
  - realized sampler composition is logged to the persistent .log file,
    starting at batch 1
  - checkpoint guard path executes without error

Run from repo root:
    python -m model.scripts.run2_validation.check_smoke_end2end               # Run 2 recipe (freeze ON)
    python -m model.scripts.run2_validation.check_smoke_end2end --no-freeze   # plain pipeline
"""
import os
import subprocess
import sys

STAGE_PREFIX = 'run2_smoke'
ARTIFACT_TEMPLATES = [
    'model/checkpoints/{stage}_mvss_lite_ep2.pt',
    'reports/{stage}_history.csv',
    'reports/{stage}_loss_curve.png',
    'reports/{stage}_sampler_composition.log',
]


def build_cmd(stage):
    cmd = [
        sys.executable, '-m', 'model.train',
        '--datasets', 'RTM', 'MIDV500',
        '--stage-name', stage,
        '--smoke-test',
        '--use-balanced-sampler',
        '--init-weights', 'model/checkpoints/stage1_mvss_lite_ep45.pt',
    ]
    if '--no-freeze' not in sys.argv:
        cmd.append('--freeze-early-layers')
    return cmd


def main():
    stage = STAGE_PREFIX + ('' if '--no-freeze' not in sys.argv else '_plain')
    artifacts = [t.format(stage=stage) for t in ARTIFACT_TEMPLATES]
    cmd = build_cmd(stage)

    print("=== Command under test ===")
    print(' '.join(cmd))
    print("\nNOTE: this runs the real train.py for 2 epochs x 2 batches (4 steps). "
          "It is a wiring check, not a training run.\n")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout
    print("=== train.py stdout (verbatim) ===")
    print(out)
    if proc.stderr:
        print("=== stderr ===")
        print(proc.stderr[-3000:])
    print(f"=== exit code: {proc.returncode} ===")

    checks = {}

    def has(needle):
        return needle in out

    checks['device line'] = has('Using device:')
    checks['pos_weight scan fired'] = has('Scanning dataset') and has('Global Stats -> Seg pos_weight:') and has('Edge pos_weight:')
    if '--no-freeze' in sys.argv:
        checks['freeze report printed'] = True  # N/A: freeze OFF by design for this variant
    else:
        checks['freeze report printed'] = has('Freeze report') and has('TOTAL::frozen') and has('Optimizer built over TRAINABLE params only')
    checks['manifest splits loaded'] = has('Dataset splits -> Train:')
    checks['sampler log announced'] = has('_sampler_composition.log')
    checks['composition logged batch1'] = '[sampler] epoch=1 batch=1' in out
    checks['epoch summaries present'] = has('=== Epoch 1 Summary ===') and has('=== Epoch 2 Summary ===')

    log_path = f'reports/{stage}_sampler_composition.log'
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read()
        lines = [l for l in content.splitlines() if l.strip()]
        print(f"\n=== persistent sampler log ({len(lines)} lines) ===")
        for l in lines[:5]:
            print(l)
        checks['persistent log exists w/ batch1'] = any('batch=1 ' in l for l in lines)
    else:
        checks['persistent log exists w/ batch1'] = False

    ckpt_path = artifacts[0]
    checks['checkpoint written by guard path'] = os.path.exists(ckpt_path)

    print("\n=== Check summary ===")
    all_ok = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    # Cleanup: delete every artifact this smoke run created so the repo stays clean
    print("\n=== Cleanup ===")
    for p in artifacts:
        if os.path.exists(p):
            os.remove(p)
            print(f"removed {p}")

    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")


if __name__ == '__main__':
    main()
