import os
import sys
import glob
import json
import argparse
import numpy as np


def compute_mean_std(values):
    cleaned = [v for v in values if v is not None and not np.isnan(v)]
    if not cleaned:
        return {'mean': None, 'std': None, 'count': 0}
    arr = np.array(cleaned, dtype=float)
    return {
        'mean': round(float(np.mean(arr)), 4),
        'std': round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
        'count': int(len(arr))
    }


def find_files(pattern_or_paths):
    found = []
    if isinstance(pattern_or_paths, str):
        candidates = glob.glob(pattern_or_paths)
    else:
        candidates = pattern_or_paths
    for path in candidates:
        if os.path.exists(path):
            found.append(path)
    return sorted(list(set(found)))


def aggregate_baserate(reports_dir):
    pattern = os.path.join(reports_dir, "baserate_experiment_run4_seed*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.stderr.write(f"WARNING: No baserate files found matching {pattern}\n")
        return None

    print(f"Loading baserate files: {files}")
    loaded = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                loaded.append((f, data))
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed loading {f}: {e}\n")

    if not loaded:
        return None

    per_condition_map = {}
    analysis_stats = {
        'pearson_r': [],
        'spearman_rho': [],
        'spearman_p': []
    }

    for f, d in loaded:
        conditions = d.get('conditions', [])
        for idx, cond in enumerate(conditions):
            br = cond.get('base_rate', idx)
            if br not in per_condition_map:
                per_condition_map[br] = {
                    'base_rate': br,
                    'auth_fp': [],
                    'forged_recall': [],
                    'n_forged': [],
                    'n_auth': []
                }
            for k in ['auth_fp', 'forged_recall', 'n_forged', 'n_auth']:
                val = cond.get(k)
                if val is not None:
                    per_condition_map[br][k].append(val)

        analysis = d.get('analysis', {})
        for k in ['pearson_r', 'spearman_rho', 'spearman_p']:
            val = analysis.get(k)
            if val is not None:
                analysis_stats[k].append(val)

    aggregated_conditions = []
    for br in sorted(per_condition_map.keys()):
        raw = per_condition_map[br]
        aggregated_conditions.append({
            'base_rate': br,
            'auth_fp': compute_mean_std(raw['auth_fp']),
            'forged_recall': compute_mean_std(raw['forged_recall']),
            'n_forged': compute_mean_std(raw['n_forged']),
            'n_auth': compute_mean_std(raw['n_auth']),
        })

    aggregated_analysis = {
        'pearson_r': compute_mean_std(analysis_stats['pearson_r']),
        'spearman_rho': compute_mean_std(analysis_stats['spearman_rho']),
        'spearman_p': compute_mean_std(analysis_stats['spearman_p']),
    }

    return {
        'files': [f for f, _ in loaded],
        'conditions': aggregated_conditions,
        'analysis': aggregated_analysis
    }


def aggregate_dethead(reports_dir):
    bal_pattern = os.path.join(reports_dir, "detection_head_run4_seed*.json")
    unbal_pattern = os.path.join(reports_dir, "detection_head_run4_unbal_seed*.json")

    unbal_files = set(glob.glob(unbal_pattern))
    bal_files = set(glob.glob(bal_pattern)) - unbal_files

    if not bal_files and not unbal_files:
        sys.stderr.write("WARNING: No detection head files found (balanced or unbalanced)\n")
        return None

    def parse_seed(path, suffix_tag="seed"):
        base = os.path.basename(path).replace(".json", "")
        parts = base.split(suffix_tag)
        if len(parts) > 1:
            return parts[-1]
        return base

    bal_by_seed = {}
    for f in bal_files:
        seed = parse_seed(f, "seed")
        try:
            with open(f, 'r') as fp:
                bal_by_seed[seed] = (f, json.load(fp))
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed loading {f}: {e}\n")

    unbal_by_seed = {}
    for f in unbal_files:
        seed = parse_seed(f, "seed")
        try:
            with open(f, 'r') as fp:
                unbal_by_seed[seed] = (f, json.load(fp))
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed loading {f}: {e}\n")

    print(f"Detection head found seeds: balanced={list(bal_by_seed.keys())}, unbalanced={list(unbal_by_seed.keys())}")

    domains = set()
    readouts = set()

    def inspect_and_collect(by_seed):
        for s, (path, data) in by_seed.items():
            per_domain = data.get('per_domain')
            if per_domain is None:
                sys.stderr.write(f"WARNING: Missing 'per_domain' in {path}. Top keys: {list(data.keys())}\n")
                continue
            for d, rdict in per_domain.items():
                domains.add(d)
                for r in rdict.keys():
                    readouts.add(r)

    inspect_and_collect(bal_by_seed)
    inspect_and_collect(unbal_by_seed)

    sorted_domains = sorted(list(domains))
    sorted_readouts = sorted(list(readouts))

    def collect_metrics(by_seed):
        res = {d: {r: {'auc': [], 'auth_fp': [], 'recall': []} for r in sorted_readouts} for d in sorted_domains}
        for s, (path, data) in by_seed.items():
            per_domain = data.get('per_domain', {})
            for d in sorted_domains:
                d_entry = per_domain.get(d, {})
                for r in sorted_readouts:
                    r_entry = d_entry.get(r, {})
                    auc = r_entry.get('auc')
                    afp = r_entry.get('auth_fp')
                    rec = r_entry.get('recall')
                    if auc is not None:
                        res[d][r]['auc'].append(auc)
                    if afp is not None:
                        res[d][r]['auth_fp'].append(afp)
                    if rec is not None:
                        res[d][r]['recall'].append(rec)
        aggregated = {d: {} for d in sorted_domains}
        for d in sorted_domains:
            for r in sorted_readouts:
                aggregated[d][r] = {
                    'auc': compute_mean_std(res[d][r]['auc']),
                    'auth_fp': compute_mean_std(res[d][r]['auth_fp']),
                    'recall': compute_mean_std(res[d][r]['recall']),
                }
        return aggregated

    bal_agg = collect_metrics(bal_by_seed)
    unbal_agg = collect_metrics(unbal_by_seed)

    paired_seeds = sorted(list(set(bal_by_seed.keys()).intersection(set(unbal_by_seed.keys()))))
    paired_deltas_raw = {d: {r: {'delta_auc': [], 'delta_auth_fp': []} for r in sorted_readouts} for d in sorted_domains}

    for s in paired_seeds:
        _, b_data = bal_by_seed[s]
        _, u_data = unbal_by_seed[s]
        b_pd = b_data.get('per_domain', {})
        u_pd = u_data.get('per_domain', {})
        for d in sorted_domains:
            for r in sorted_readouts:
                b_entry = b_pd.get(d, {}).get(r, {})
                u_entry = u_pd.get(d, {}).get(r, {})
                b_auc = b_entry.get('auc')
                u_auc = u_entry.get('auc')
                b_afp = b_entry.get('auth_fp')
                u_afp = u_entry.get('auth_fp')
                if b_auc is not None and u_auc is not None:
                    paired_deltas_raw[d][r]['delta_auc'].append(b_auc - u_auc)
                if b_afp is not None and u_afp is not None:
                    paired_deltas_raw[d][r]['delta_auth_fp'].append(b_afp - u_afp)

    paired_deltas_agg = {d: {} for d in sorted_domains}
    for d in sorted_domains:
        for r in sorted_readouts:
            paired_deltas_agg[d][r] = {
                'delta_auc': compute_mean_std(paired_deltas_raw[d][r]['delta_auc']),
                'delta_auth_fp': compute_mean_std(paired_deltas_raw[d][r]['delta_auth_fp']),
            }

    return {
        'balanced_files': [f for f, _ in bal_by_seed.values()],
        'unbalanced_files': [f for f, _ in unbal_by_seed.values()],
        'paired_seeds': paired_seeds,
        'balanced': bal_agg,
        'unbalanced': unbal_agg,
        'paired_deltas': paired_deltas_agg
    }


def aggregate_calibration(reports_dir):
    pattern = os.path.join(reports_dir, "calibration_baseline_seed*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.stderr.write(f"WARNING: No calibration files found matching {pattern}\n")
        return None

    print(f"Loading calibration files: {files}")
    loaded = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                loaded.append((f, json.load(fp)))
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed loading {f}: {e}\n")

    if not loaded:
        return None

    readouts = set()
    domains = set()
    for f, d in loaded:
        per_readout = d.get('per_readout')
        if per_readout is None:
            sys.stderr.write(f"WARNING: Missing 'per_readout' in {f}. Top keys: {list(d.keys())}\n")
            continue
        for r, rdata in per_readout.items():
            readouts.add(r)
            for variant in ['uncalibrated', 'temperature_calibrated', 'isotonic_calibrated']:
                vdict = rdata.get(variant, {})
                for dom in vdict.keys():
                    domains.add(dom)

    sorted_readouts = sorted(list(readouts))
    sorted_domains = sorted(list(domains))
    variants = ['uncalibrated', 'temperature_calibrated', 'isotonic_calibrated']

    collated = {
        r: {
            v: {
                d: {'auth_fp': [], 'ece': [], 'auc': [], 'recall': []}
                for d in sorted_domains
            }
            for v in variants
        }
        for r in sorted_readouts
    }

    opt_temps = {r: [] for r in sorted_readouts}

    for f, d in loaded:
        per_readout = d.get('per_readout', {})
        for r in sorted_readouts:
            rdata = per_readout.get(r, {})
            t = rdata.get('optimal_temperature')
            if t is not None:
                opt_temps[r].append(t)
            for v in variants:
                vdict = rdata.get(v, {})
                for dom in sorted_domains:
                    metrics = vdict.get(dom, {})
                    for m in ['auth_fp', 'ece', 'auc', 'recall']:
                        val = metrics.get(m)
                        if val is not None:
                            collated[r][v][dom][m].append(val)

    aggregated = {
        r: {
            'optimal_temperature': compute_mean_std(opt_temps[r]),
            'variants': {
                v: {
                    dom: {
                        m: compute_mean_std(collated[r][v][dom][m])
                        for m in ['auth_fp', 'ece', 'auc', 'recall']
                    }
                    for dom in sorted_domains
                }
                for v in variants
            }
        }
        for r in sorted_readouts
    }

    return {
        'files': [f for f, _ in loaded],
        'per_readout': aggregated
    }


def print_summary(results):
    print("\n" + "=" * 70)
    print("MULTISEED AGGREGATION SUMMARY")
    print("=" * 70)

    br = results.get('baserate')
    if br:
        print("\n--- [1] BASE-RATE EXPERIMENT ---")
        print(f"Files included ({len(br['files'])}): {br['files']}")
        print(f"{'Base Rate':>10s} | {'Authentic FP':>18s} | {'Forged Recall':>18s}")
        for c in br['conditions']:
            b_val = c['base_rate']
            fp_m, fp_s = c['auth_fp']['mean'], c['auth_fp']['std']
            rec_m, rec_s = c['forged_recall']['mean'], c['forged_recall']['std']
            print(f"{b_val:>10.2f} | {fp_m:.4f} +/- {fp_s:.4f}  | {rec_m:.4f} +/- {rec_s:.4f}")
        ana = br['analysis']
        print(f"Spearman rho : {ana['spearman_rho']['mean']} +/- {ana['spearman_rho']['std']}")
        print(f"Spearman p   : {ana['spearman_p']['mean']} +/- {ana['spearman_p']['std']}")
        print(f"Pearson r    : {ana['pearson_r']['mean']} +/- {ana['pearson_r']['std']}")

    dh = results.get('detection_head')
    if dh:
        print("\n--- [2] DETECTION HEAD (BALANCED vs UNBALANCED) ---")
        print(f"Balanced files ({len(dh['balanced_files'])}), Unbalanced files ({len(dh['unbalanced_files'])})")
        print(f"Paired seeds for deltas ({len(dh['paired_seeds'])}): {dh['paired_seeds']}")
        domains = sorted(list(dh['balanced'].keys())) if dh['balanced'] else []
        for d in domains:
            print(f"\nDomain: {d}")
            print(f"{'Readout':>14s} | {'Bal AUC':>16s} | {'Unbal AUC':>16s} | {'Delta AUC':>16s} | {'Bal FP':>16s} | {'Unbal FP':>16s} | {'Delta FP':>16s}")
            readouts = sorted(list(dh['balanced'][d].keys()))
            for r in readouts:
                b_auc = dh['balanced'].get(d, {}).get(r, {}).get('auc', {})
                u_auc = dh['unbalanced'].get(d, {}).get(r, {}).get('auc', {})
                d_auc = dh['paired_deltas'].get(d, {}).get(r, {}).get('delta_auc', {})

                b_fp = dh['balanced'].get(d, {}).get(r, {}).get('auth_fp', {})
                u_fp = dh['unbalanced'].get(d, {}).get(r, {}).get('auth_fp', {})
                d_fp = dh['paired_deltas'].get(d, {}).get(r, {}).get('delta_auth_fp', {})

                def fmt(m):
                    if m.get('mean') is None:
                        return "N/A"
                    return f"{m['mean']:.4f}+/-{m['std']:.4f}"

                print(f"{r:>14s} | {fmt(b_auc):>16s} | {fmt(u_auc):>16s} | {fmt(d_auc):>16s} | {fmt(b_fp):>16s} | {fmt(u_fp):>16s} | {fmt(d_fp):>16s}")

    cal = results.get('calibration')
    if cal:
        print("\n--- [3] CALIBRATION BASELINE (BEFORE vs AFTER) ---")
        print(f"Files included ({len(cal['files'])}): {cal['files']}")
        for r, rdata in cal['per_readout'].items():
            t_m = rdata['optimal_temperature']['mean']
            t_s = rdata['optimal_temperature']['std']
            print(f"\nReadout: {r} (Optimal T = {t_m} +/- {t_s})")
            variants = rdata['variants']
            domains = list(variants.get('uncalibrated', {}).keys())
            print(f"{'Domain':>10s} | {'Variant':>15s} | {'FP@match-recall':>18s} | {'ECE':>18s} | {'AUC':>18s}")
            for d in domains:
                for vname, vkey in [('Uncalibrated', 'uncalibrated'),
                                    ('Temperature', 'temperature_calibrated'),
                                    ('Isotonic', 'isotonic_calibrated')]:
                    mdict = variants.get(vkey, {}).get(d, {})
                    def fmt(m):
                        if m.get('mean') is None:
                            return "N/A"
                        return f"{m['mean']:.4f}+/-{m['std']:.4f}"
                    print(f"{d:>10s} | {vname:>15s} | {fmt(mdict.get('auth_fp', {})):>18s} | {fmt(mdict.get('ece', {})):>18s} | {fmt(mdict.get('auc', {})):>18s}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multiseed diagnostic runs.")
    parser.add_argument('--reports-dir', default='reports', help="Directory containing report JSONs (default: reports)")
    parser.add_argument('--out', default='reports/aggregate_multiseed.json', help="Output aggregate JSON path")
    args = parser.parse_args()

    reports_dir = args.reports_dir
    baserate_agg = aggregate_baserate(reports_dir)
    dethead_agg = aggregate_dethead(reports_dir)
    calib_agg = aggregate_calibration(reports_dir)

    if baserate_agg is None and dethead_agg is None and calib_agg is None:
        sys.stderr.write("ERROR: No files found across any artifact family! Exiting nonzero.\n")
        sys.exit(1)

    out_data = {
        'baserate': baserate_agg,
        'detection_head': dethead_agg,
        'calibration': calib_agg
    }

    print_summary(out_data)

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, 'w') as fp:
            json.dump(out_data, fp, indent=2)
        print(f"Saved aggregated report to {args.out}")

    sys.exit(0)


if __name__ == '__main__':
    main()
