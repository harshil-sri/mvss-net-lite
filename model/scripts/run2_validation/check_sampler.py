"""
Check 3 (the critical one): realized sampler composition vs the 1/3-1/3-1/3 target.

Uses the REAL manifest-filtered train split and the REAL WeightedRandomSampler
built by get_dataloader(..., use_balanced_sampler=True). No model, no GPU, no
image decoding - we iterate the sampler's index stream directly (exactly what
DataLoader does each epoch), bucket draws into batches of BATCH_SIZE, and count
realized composition.

Also runs the OLD hardcoded-weight scheme (1/6000, 1/3000, 1/15050) on the same
measured population so the before/after difference is raw data, not a claim.

Run from repo root:
    python -m model.scripts.run2_validation.check_sampler
"""
from collections import Counter

from data_pipeline.dataset_loader import get_dataloader

BATCH_SIZE = 8          # matches train.py BATCH_SIZE
EPOCHS_TO_SIMULATE = 5  # sampler draws per "epoch" == len(dataset), like real training
TOLERANCE_PP = 2.0      # acceptable deviation from 33.33%, in percentage points


def simulate_batches(sampler, n_epochs, batch_size):
    """Draws indices exactly as DataLoader would across epochs; returns list of batches."""
    batches = []
    for _ in range(n_epochs):
        idx_iter = iter(sampler)
        buf = []
        for idx in idx_iter:
            buf.append(idx)
            if len(buf) == batch_size:
                batches.append(buf)
                buf = []
        if buf:  # partial trailing batch (DataLoader emits it too)
            batches.append(buf)
    return batches


def composition_stats(batches, classes):
    total = Counter()
    per_batch_pct = {c: [] for c in ('rtm_forged', 'rtm_auth', 'midv500')}
    for b in batches:
        c = Counter(classes[i] for i in b)
        total.update(c)
        n = len(b)
        for cls in per_batch_pct:
            per_batch_pct[cls].append(100.0 * c.get(cls, 0) / n)
    return total, per_batch_pct


def report(tag, total, per_batch_pct, n_batches):
    grand = sum(total.values())
    print(f"\n--- {tag} ---")
    print(f"batches simulated: {n_batches} | total draws: {grand}")
    ok = True
    for cls in ('rtm_forged', 'rtm_auth', 'midv500'):
        share = 100.0 * total[cls] / grand
        pcts = per_batch_pct[cls]
        mean_pb = sum(pcts) / len(pcts)
        mn, mx = min(pcts), max(pcts)
        dev = abs(share - 100.0 / 3)
        within = dev <= TOLERANCE_PP
        ok = ok and within
        print(f"{cls:12s}: overall={share:6.2f}% | target=33.33% | dev={dev:4.2f}pp | "
              f"per-batch mean={mean_pb:6.2f}% min={mn:5.2f}% max={mx:5.2f}% | "
              f"within ±{TOLERANCE_PP}pp: {'YES' if within else 'NO'}")
    other = grand - sum(total[c] for c in per_batch_pct)
    print(f"other       : {other} draws")
    return ok


def repetition_analysis(sampler, classes, n_epochs):
    """
    Per-epoch exposure stats: with replacement sampling, rare classes are
    oversampled, so individual images repeat within an epoch. Quantify it:
    draws-per-image distribution per class, unique coverage, never-drawn share.
    """
    from collections import Counter as C
    pool = C(classes)
    print(f"\n=== Repetition analysis ({n_epochs} epochs x {sampler.num_samples} draws/epoch) ===")
    cum = {c: C() for c in ('rtm_forged', 'rtm_auth', 'midv500')}
    for ep in range(1, n_epochs + 1):
        counts = {c: C() for c in cum}
        for idx in iter(sampler):
            c = classes[idx]
            if c in counts:
                counts[c][idx] += 1
                cum[c][idx] += 1
        for c in ('rtm_forged', 'rtm_auth', 'midv500'):
            d = counts[c]
            total_draws = sum(d.values())
            uniq = len(d)
            mean_rep = total_draws / uniq if uniq else 0
            max_rep = max(d.values()) if d else 0
            never = pool.get(c, 0) - uniq
            never_pct = 100.0 * never / pool.get(c, 1)
            print(f"epoch {ep} {c:12s}: draws={total_draws} | unique_images={uniq}/{pool.get(c,0)} "
                  f"| mean_repeats_per_drawn_img={mean_rep:.2f} | max_repeats={max_rep} "
                  f"| never_drawn={never} ({never_pct:.1f}%)")
    print("cumulative across epochs:")
    for c in ('rtm_forged', 'rtm_auth', 'midv500'):
        d = cum[c]
        total_draws = sum(d.values())
        uniq = len(d)
        mean_overall = total_draws / pool.get(c, 1)
        max_total = max(d.values()) if d else 0
        print(f"  {c:12s}: total_draws={total_draws} | unique_images={uniq}/{pool.get(c,0)} "
              f"| mean_epoch_exposures_per_image={mean_overall:.2f} | max_total_repeats={max_total}")


def main():
    print("Building REAL train split via get_dataloader (manifest-based)...")
    train_loader, _, _ = get_dataloader(
        ['RTM', 'MIDV500'], batch_size=BATCH_SIZE, is_train=True,
        return_splits=True, use_balanced_sampler=True)

    ds = train_loader.dataset
    base = ds.base if hasattr(ds, 'base') else ds
    classes = base.sample_classes
    pop = Counter(classes)

    print("\n=== Measured train-split population (ground truth from masks) ===")
    for k, v in sorted(pop.items()):
        pct = 100.0 * v / len(classes)
        print(f"{k:12s}: {v} ({pct:.2f}% of pool)")

    sampler_new = train_loader.sampler
    print(f"\nNew dynamic sampler: num_samples={sampler_new.num_samples}, "
          f"replacement={sampler_new.replacement}")

    # --- NEW scheme, empirical ---
    batches = simulate_batches(sampler_new, EPOCHS_TO_SIMULATE, BATCH_SIZE)
    total_new, pb_new = composition_stats(batches, classes)
    ok_new = report(f"NEW dynamic weights ({EPOCHS_TO_SIMULATE} epochs x {len(batches)//EPOCHS_TO_SIMULATE} batches)",
                    total_new, pb_new, len(batches))

    # --- OLD scheme, empirical (same population, hardcoded denominators) ---
    old_weights = []
    for lab in classes:
        if lab == 'rtm_forged':
            old_weights.append(1.0 / 6000)
        elif lab == 'rtm_auth':
            old_weights.append(1.0 / 3000)
        elif lab == 'midv500':
            old_weights.append(1.0 / 15050)
        else:
            old_weights.append(1.0)
    from torch.utils.data import WeightedRandomSampler
    sampler_old = WeightedRandomSampler(old_weights, num_samples=len(old_weights), replacement=True)
    batches_old = simulate_batches(sampler_old, EPOCHS_TO_SIMULATE, BATCH_SIZE)
    total_old, pb_old = composition_stats(batches_old, classes)
    report(f"OLD hardcoded weights (6000/3000/15050) - same population", total_old, pb_old, len(batches_old))

    print("\n=== First 10 NEW-scheme batches (raw per-batch counts) ===")
    for i, b in enumerate(batches[:10]):
        c = Counter(classes[j] for j in b)
        print(f"batch {i+1}: rtm_forged={c.get('rtm_forged',0)} "
              f"rtm_auth={c.get('rtm_auth',0)} midv500={c.get('midv500',0)}")

    repetition_analysis(sampler_new, classes, EPOCHS_TO_SIMULATE)

    print(f"\nRESULT: {'PASS' if ok_new else 'FAIL'} "
          f"(new-scheme overall shares within ±{TOLERANCE_PP}pp of 33.33%)")


if __name__ == '__main__':
    main()
