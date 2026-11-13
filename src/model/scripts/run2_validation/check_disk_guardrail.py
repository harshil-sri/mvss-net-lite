"""
Check 5: checkpoint disk guardrail + rotation, unit-tested with mocked disk
space. No epochs are run; no real training occurs.

Covers:
  A) Low free space -> save_checkpoint_with_guard returns False, writes
     reports/DISK_FULL_WARNING.txt, does NOT create the checkpoint, does NOT
     raise (graceful halt is the caller's decision - exactly what train.py does).
  B) Sufficient space -> returns True and writes the file.
  C) rotate_checkpoints keeps at most `keep` newest, NEVER deletes the current
     best-validation checkpoint even when it is the oldest.

Run from repo root:
    python -m model.scripts.run2_validation.check_disk_guardrail
"""
import os
import tempfile
from collections import namedtuple
from unittest import mock

import torch

from model.train import save_checkpoint_with_guard, rotate_checkpoints

_usage_nt = namedtuple('usage', ['total', 'used', 'free'])


def fake_disk_usage(free_bytes):
    total = 500 * 1024**3
    used = total - free_bytes
    return _usage_nt(total=total, used=used, free=free_bytes)


def case_a_low_space(tmp):
    print("\n--- Case A: low free space (10GB < 15GB required) ---")
    target = os.path.join(tmp, 'never_written.pt')
    warn_dir = os.path.join(tmp, 'reports')
    with mock.patch('shutil.disk_usage', return_value=fake_disk_usage(10 * 1024**3)):
        # NOTE: train.py calls shutil.disk_usage via the module-level import in
        # model.train; patching shutil.disk_usage globally covers both.
        ok, free_gb = save_checkpoint_with_guard(
            {'model_state_dict': {}}, target,
            warning_dir=warn_dir, context_msg='(unit-test)')
    print(f"returned ok={ok} | reported_free={free_gb:.2f}GB")
    warn_file = os.path.join(warn_dir, 'DISK_FULL_WARNING.txt')
    wrote_warning = os.path.exists(warn_file)
    created_ckpt = os.path.exists(target)
    if wrote_warning:
        with open(warn_file) as f:
            print(f"warning file content: {f.read().strip()}")
    print(f"warning file written: {wrote_warning} | checkpoint created: {created_ckpt} "
          f"(reached this line == no exception propagated)")
    passed = (ok is False) and wrote_warning and (not created_ckpt)
    return passed


def case_b_ok_space(tmp):
    print("\n--- Case B: sufficient free space (100GB) ---")
    payload = {'model_state_dict': {'k': torch.zeros(3)}}
    target = os.path.join(tmp, 'should_exist.pt')
    with mock.patch('shutil.disk_usage', return_value=fake_disk_usage(100 * 1024**3)):
        ok, free_gb = save_checkpoint_with_guard(payload, target, warning_dir=os.path.join(tmp, 'r2'))
    exists = os.path.exists(target)
    reloaded = torch.load(target, weights_only=False) if exists else None
    roundtrip = bool(reloaded and reloaded['model_state_dict']['k'].equal(torch.zeros(3)))
    print(f"returned ok={ok} | file exists={exists} | reload roundtrip ok={roundtrip}")
    return ok and exists and roundtrip


def case_c_rotation(tmp):
    print("\n--- Case C: rotation keeps <=3, never deletes best ---")
    paths = []
    for i in range(5):
        p = os.path.join(tmp, f'ep{i}.pt')
        torch.save({'i': i}, p)
        paths.append(p)
    best_is_oldest = paths[0]  # adversarial: best checkpoint is also the oldest

    removed1 = rotate_checkpoints(paths, best_is_oldest, keep=3)
    print(f"after rotate#1: remaining={[os.path.basename(p) for p in paths]} removed={[os.path.basename(p) for p in removed1]}")
    cond1 = len(paths) == 3 and best_is_oldest not in removed1 and os.path.exists(best_is_oldest)

    removed2 = rotate_checkpoints(paths, None, keep=3)
    print(f"after rotate#2 (no best tracked): remaining={[os.path.basename(p) for p in paths]} "
          f"removed={[os.path.basename(p) for p in removed2]}")
    cond2 = len(paths) == 0 or all(not os.path.exists(p) for p in removed2)

    over = [f'x{i}.pt' for i in range(2)]  # fewer than keep -> no-op
    removed3 = rotate_checkpoints(over, None, keep=3)
    cond3 = len(over) == 2 and removed3 == []
    print(f"under-keep no-op check: list unchanged={cond3}")
    return cond1 and cond2 and cond3


def main():
    tmp = tempfile.mkdtemp(prefix='disk_guard_', dir='/tmp/opencode')
    a = case_a_low_space(tmp)
    b = case_b_ok_space(tmp)
    c = case_c_rotation(tmp)
    print("\n=== Summary ===")
    print(f"A graceful halt on low space : {'PASS' if a else 'FAIL'}")
    print(f"B normal save works          : {'PASS' if b else 'FAIL'}")
    print(f"C rotation + best protection : {'PASS' if c else 'FAIL'}")
    print(f"\nRESULT: {'PASS' if (a and b and c) else 'FAIL'}")


if __name__ == '__main__':
    main()
