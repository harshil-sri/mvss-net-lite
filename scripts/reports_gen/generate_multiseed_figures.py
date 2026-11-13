import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Style parameters matching repo convention (figstyle.py / generate_figures.py)
COL_W = 3.4
FULL_W = 7.1

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
GREY = "#8C8C8C"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.alpha": 0.25,
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})

def save_fig(fig, path):
    target = str(path) if str(path).endswith(".pdf") else f"{path}.pdf"
    fig.savefig(target, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def generate_fig_sweep(out_path):
    seeds = [42, 123, 7]
    sweep_data = {}
    for s in seeds:
        with open(f"src/reports/baserate_experiment_run4_seed{s}.json") as f:
            sweep_data[s] = json.load(f)["conditions"]

    base_rates = np.array([c["base_rate"] for c in sweep_data[42]])
    fp_matrix = np.array([[c["auth_fp"] for c in sweep_data[s]] for s in seeds])
    rec_matrix = np.array([[c["forged_recall"] for c in sweep_data[s]] for s in seeds])

    fp_mean = np.mean(fp_matrix, axis=0)
    fp_std = np.std(fp_matrix, axis=0, ddof=1)
    rec_mean = np.mean(rec_matrix, axis=0)
    rec_std = np.std(rec_matrix, axis=0, ddof=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, COL_W))

    ax1.plot(base_rates, fp_mean, marker="o", linestyle="-", color=BLUE, label="Authentic FP")
    ax1.fill_between(base_rates, fp_mean - fp_std, fp_mean + fp_std, color=BLUE, alpha=0.2)
    ax1.errorbar(base_rates, fp_mean, yerr=fp_std, fmt="none", ecolor=BLUE, capsize=3, capthick=0.8, elinewidth=1.0)

    ax1.plot(base_rates, rec_mean, marker="s", linestyle="--", color=ORANGE, label="Forged Recall")
    ax1.fill_between(base_rates, rec_mean - rec_std, rec_mean + rec_std, color=ORANGE, alpha=0.2)
    ax1.errorbar(base_rates, rec_mean, yerr=rec_std, fmt="none", ecolor=ORANGE, capsize=3, capthick=0.8, elinewidth=1.0)

    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Base rate p")
    ax1.set_ylabel("Rate")
    ax1.legend(frameon=False, loc="center left")
    ax1.grid(True, alpha=0.25)
    ax1.text(-0.1, 1.05, "(a)", transform=ax1.transAxes, fontsize=8, fontweight="bold", va="top")

    metrics = ["Authentic FP", "Forged Recall"]
    p0_fp = (fp_mean[0], fp_std[0])
    p1_fp = (fp_mean[-1], fp_std[-1])
    p0_rec = (rec_mean[0], rec_std[0])
    p1_rec = (rec_mean[-1], rec_std[-1])

    pairs = [
        (p0_fp, p1_fp, BLUE),
        (p0_rec, p1_rec, ORANGE)
    ]

    for i, (p0_val, p1_val, col) in enumerate(pairs):
        ax2.plot([p0_val[0], p1_val[0]], [i, i], color=col, linewidth=2, marker="o", markersize=5)
        ax2.errorbar([p0_val[0], p1_val[0]], [i, i], xerr=[p0_val[1], p1_val[1]], fmt="none", ecolor=col, capsize=3, capthick=0.8, elinewidth=1.0)
        ax2.text(p0_val[0] - 0.03, i, f"{p0_val[0]:.2f}", ha="right", va="center", fontsize=7)
        ax2.text(p1_val[0] + 0.03, i, f"{p1_val[0]:.2f}", ha="left", va="center", fontsize=7)

    ax2.set_xlim(-0.08, 1.15)
    ax2.set_yticks(range(len(metrics)))
    ax2.set_yticklabels(metrics)
    ax2.set_xlabel("Rate at p=0 vs p=1")
    ax2.grid(True, alpha=0.25)
    ax2.text(-0.1, 1.05, "(b)", transform=ax2.transAxes, fontsize=8, fontweight="bold", va="top")

    plt.tight_layout()
    save_fig(fig, out_path)

def generate_fig_dethead(out_path):
    seeds = [42, 123, 7]
    balanced = {s: json.load(open(f"src/reports/detection_head_run4_seed{s}.json"))["per_domain"] for s in seeds}
    unbalanced = {s: json.load(open(f"src/reports/detection_head_run4_unbal_seed{s}.json"))["per_domain"] for s in seeds}

    readout_keys = ["segmean", "segtopk", "dethead_mean", "dethead_max"]
    readout_labels = ["seg-mean", "top-$k$", "det-mean", "det-max"]

    auc_casia_mean = [np.mean([balanced[s]["CASIAv2"][k]["auc"] for s in seeds]) for k in readout_keys]
    auc_casia_std = [np.std([balanced[s]["CASIAv2"][k]["auc"] for s in seeds], ddof=1) for k in readout_keys]
    auc_rtm_mean = [np.mean([balanced[s]["RTM"][k]["auc"] for s in seeds]) for k in readout_keys]
    auc_rtm_std = [np.std([balanced[s]["RTM"][k]["auc"] for s in seeds], ddof=1) for k in readout_keys]

    fp_casia_mean = [np.mean([balanced[s]["CASIAv2"][k]["auth_fp"] for s in seeds]) for k in readout_keys]
    fp_casia_std = [np.std([balanced[s]["CASIAv2"][k]["auth_fp"] for s in seeds], ddof=1) for k in readout_keys]
    fp_rtm_mean = [np.mean([balanced[s]["RTM"][k]["auth_fp"] for s in seeds]) for k in readout_keys]
    fp_rtm_std = [np.std([balanced[s]["RTM"][k]["auth_fp"] for s in seeds], ddof=1) for k in readout_keys]

    x = np.arange(len(readout_keys))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 2.6), sharex=True, constrained_layout=True)

    ax1.bar(x - w / 2, auc_casia_mean, w, yerr=auc_casia_std, capsize=3, label="CASIAv2", color=BLUE, error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax1.bar(x + w / 2, auc_rtm_mean, w, yerr=auc_rtm_std, capsize=3, label="RTM", color=ORANGE, error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax1.set_ylim(0.70, 0.90)
    ax1.set_ylabel("AUC")
    ax1.text(0.02, 0.92, "(a)", transform=ax1.transAxes, fontweight="bold", va="top")
    ax1.legend(loc="lower right", ncol=2, frameon=False)

    ax2.bar(x - w / 2, fp_casia_mean, w, yerr=fp_casia_std, capsize=3, label="CASIAv2", color=BLUE, error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax2.bar(x + w / 2, fp_rtm_mean, w, yerr=fp_rtm_std, capsize=3, label="RTM", color=ORANGE, error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax2.set_ylim(0, 0.55)
    ax2.set_ylabel("Authentic FP")
    ax2.text(0.02, 0.92, "(b)", transform=ax2.transAxes, fontweight="bold", va="top")
    ax2.set_xticks(x)
    ax2.set_xticklabels(readout_labels)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--")

    save_fig(fig, out_path)

if __name__ == "__main__":
    out_dir = Path("workspace/final version/paper/figures_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_fig_sweep(out_dir / "fig_sweep.pdf")
    generate_fig_dethead(out_dir / "fig_dethead.pdf")
    print("Done")
