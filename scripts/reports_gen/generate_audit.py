import json
import os

out_dir = "/home/harshil/mvss-net-lite/audit_dossier"
os.makedirs(out_dir, exist_ok=True)

# 00_EXECUTIVE_SUMMARY.md
with open(os.path.join(out_dir, "00_EXECUTIVE_SUMMARY.md"), "w") as f:
    f.write("""# Executive Summary: Research Audit Dossier

**Project:** MVSS-Net-Lite
**Date of Audit:** October 7, 2026
**Target:** Candidate Final Paper (Version 2)

## Overview
This dossier contains a comprehensive, brutal, and forensically accurate audit of the MVSS-Net-Lite project. The goal is to provide a final reviewer with an unvarnished assessment of the paper's claims versus the repository's artifacts. 

## Key Findings
1. **Core Contribution (Supported):** The causal relationship between training-set forgery base rate and authentic-image false positive rate (FPR) is strongly supported. The base-rate sweep ($\rho=1.0$) demonstrates that the decision threshold, not learned semantic features, dominates cross-domain failure.
2. **Architecture (Partially Supported):** MVSS-Net-Lite is implemented as a Dual ResNet-34 with BayarConv noise extraction and CBAM fusion. While lighter than a Dual ResNet-50, it is not deeply "lightweight" (it still has two massive ResNet-34 branches).
3. **Statistical Weakness:** The primary correlation relies on 5 data points ($b \in \{0, 0.25, 0.5, 0.75, 1.0\}$) for the causal sweep. Some mitigation experiments (e.g., decorrelation) failed to replicate across random seeds (seed-42 "win" was a fluke).
4. **Contradictions:** The original hypothesis that "breaking the domain-label correlation fixes the generalization problem" was CONTRADICTED by the repository's own experiments. Group-DRO failed completely (FPR 1.00 at 85% recall) and DANN provided no meaningful improvement over ERM.
5. **Reviewer Attack Vector:** Reviewers will heavily attack the claim of "domain generalization" because the paper's actual finding is a *negative* result: standard DG methods (DANN/DRO) fail, and the problem is fundamentally a base-rate/prior shift issue that is heavily reliant on thresholding.

**Conclusion:** The paper is PAPER-READY WITH REVISIONS. The negative result is powerful, but any claims of "solving" the domain shift or proposing a new state-of-the-art mitigation must be ruthlessly excised in favor of the diagnostic/causal findings.
""")

# 01_REPOSITORY_MAP.md
with open(os.path.join(out_dir, "01_REPOSITORY_MAP.md"), "w") as f:
    f.write("""# Repository Map & Reconnaissance

## Structure
- `model/` (contains `network.py`, `backbone.py`, `pathb_network.py`): Architecture definitions.
- `reports/`: JSON logs (e.g., `baserate_experiment_run4.json`), logs of experiment runs.
- `diagnostics/`: Scripts for evaluating threshold behaviors and FPR.
- `workspace/version 1/`: Original paper draft (Oct 3).
- `workspace/version 2/`: Orchestrated paper draft (Candidate final).

## Flags Found
- **TODO/FIXME:** Present in experiment scripts regarding random seeds.
- **Hardcoded Thresholds:** Found in evaluation scripts; a fixed $0.5$ threshold was frequently used instead of dynamically calibrated thresholds, heavily confounding the FPR results.
- **Baselines:** ResNet-UNet-FPN is implemented locally (`pathb_network.py`). DANN and Group-DRO logs exist but show failures.
""")

# 02_MODEL_ARCHITECTURE.md
with open(os.path.join(out_dir, "02_MODEL_ARCHITECTURE.md"), "w") as f:
    f.write("""# Actual Model Specification

1. **Backbone:** Dual ResNet-34 (IMPLEMENTED: `model/backbone.py`)
2. **Backbone variant:** Standard ResNet-34 stem and stages, frozen early layers (`model/freezing.py`).
3. **RGB branch:** Standard ResNet-34 taking RGB.
4. **Noise/residual branch:** Constrained BayarConv extraction followed by ResNet-34.
5. **Fusion mechanism:** CBAM block per stage (IMPLEMENTED: `model/network.py`).
6. **Attention mechanisms:** Spatial and Channel attention within CBAM.
7. **Decoder:** Multi-scale feature aggregation (UNet-like upsampling).
8. **Final prediction head:** Segmentation head yielding pixel-wise logits.
9. **Parameter count:** ~45M+ (Dual ResNet-34 + Decoder).
10. **Differences from MVSS-Net:** Uses ResNet-34 instead of ResNet-50. Replaces thick morphological edges with Canny edge extraction (DESCRIBED, need to verify if actively used in final loss). Fuses via CBAM instead of dense cross-attention.

**Verdict:** The architecture matches the claims in Version 2, though "lightweight" is a relative term compared to MVSS-Net-50.
""")

# 03_DATASETS.md
with open(os.path.join(out_dir, "03_DATASETS.md"), "w") as f:
    f.write("""# Dataset Audit

- **Datasets Used:** RTM (Robust Test Model / Synthetic), CASIA (implied/used for cross-domain).
- **Leakage Assessment:** 
  - *Domain Leakage:* Cross-domain tests explicitly use separated domains. However, the exact splits (which RTM variants are in train vs. test) require careful boundary checks.
  - *Pre-processing Leakage:* Image resolutions were standardized (e.g., 256x256 due to RAM limits). This aggressive downscaling fundamentally alters high-frequency noise artifacts, acting as an unintended low-pass filter. The paper MUST acknowledge this limitation.
- **Mask Generation:** Canny edge masks used for auxiliary boundary loss (if active).

**Critical Risk:** The 256x256 limit (enforced by the 7.8GB RAM constraint) compromises high-frequency forensic noise. This is documented in the project logs and must be transparent in the paper.
""")

# 04_EXPERIMENT_INVENTORY.md
with open(os.path.join(out_dir, "04_EXPERIMENT_INVENTORY.md"), "w") as f:
    f.write("""# Experiment Inventory

| ID | Purpose | Model | Setup | Result | Artifact | Status |
|---|---|---|---|---|---|---|
| EXP-01 | Base Rate Sweep | MVSS-Lite | Train with $b \in \{0, .25, .5, .75, 1.0\}$ | Auth-FPR strongly correlates with $b$ ($\rho=1.0$) | `baserate_experiment_run4.json` | Reproducible |
| EXP-02 | Random Init Control | MVSS-Lite | Untrained weights, linear probe | ~80% AUC separability on decision layer | `reports/` | Reproducible |
| EXP-03 | DG Mitigations (DANN/DRO) | MVSS/UNet | Standard DG setups | DANN FPR=0.36, DRO FPR=1.00 at 85% recall | `paper1_plan.md` logs | Failed (Negative Result) |
| EXP-04 | Decorrelation Regularizer | MVSS-Lite | Seed 42 | Looked like a win, failed on other seeds | `paper1_plan.md` notes | **Contradicted (Fluke)** |

**Note:** The decorrelation failure is explicitly noted in internal logs but must NOT be claimed as a success in the paper.
""")

# 05_BASELINES.md
with open(os.path.join(out_dir, "05_BASELINES.md"), "w") as f:
    f.write("""# Baseline Audit

1. **ResNet-UNet-FPN:** Implemented locally (`model/pathb_network.py`). Used as a contrast to dual-branch networks. Showed $\rho=0.90$ base rate correlation (compared to MVSS-Net-Lite's $\rho=1.0$).
2. **ERM (Empirical Risk Minimization):** Standard training. Yielded FPR ~0.34 at 85% recall.
3. **DANN (Domain-Adversarial Neural Network):** Yielded FPR ~0.36 at 85% recall. Not an improvement over ERM.
4. **Group DRO:** Yielded FPR = 1.00 at 85% recall. Complete collapse.

**Fairness:** Baselines appear fair as they were run under identical compute constraints (256x256, same batch size, AMP, etc.).
""")

# 06_CROSS_DOMAIN_ANALYSIS.md
with open(os.path.join(out_dir, "06_CROSS_DOMAIN_ANALYSIS.md"), "w") as f:
    f.write("""# Cross-Domain / False-Positive Investigation

This is the crown jewel of the research.

1. **Authentic-Image FPR:** Strongly driven by the training base rate.
2. **Correlation:** 
   - MVSS-Net-Lite: $\rho = 1.0$ (Spearman), $r = 0.84$ (Pearson). Auth-FP sweep: $\{0.010, 0.751, 0.825, 0.997, 1.000\}$.
   - ResNet-UNet-FPN: $\rho = 0.90$.
3. **Random Init Control:** Untrained networks still perfectly separate domains (~80% AUC), proving that architectural inductive biases cluster domains *before* any semantic learning occurs.
4. **Threshold Behavior:** The findings suggest that the network learns a "default firing prior" based on base rates, causing the decision hyperplane to shift. Using a fixed threshold ($0.5$) exposes this, whereas per-domain calibrated thresholds might hide it.
5. **Mitigations:** ERM, DANN, and Group DRO failed to fix the fundamental prior shift.

**Verdict:** The numbers (0.34, 0.36, 1.00 FPR) and correlations ($\rho=1.0$) are explicitly logged and validated by the internal `paper1_plan.md` evidence matrix.
""")

# 07_STATISTICAL_AUDIT.md
with open(os.path.join(out_dir, "07_STATISTICAL_AUDIT.md"), "w") as f:
    f.write("""# Statistical Audit

- **Sample Size (Sweep):** 5 points for the base rate sweep ($n=5$). This is extremely small for Pearson correlation ($r=0.84$), though Spearman rank correlation ($\rho=1.0$) is robust to the monotonic trend.
- **Multiple Seeds:** Internal logs explicitly warn that "Seed-42 decorrelation win did NOT replicate across seeds." The authors successfully caught this, but the final paper MUST NOT claim decorrelation works.
- **Threshold Selection:** Post-hoc threshold analysis (fixing recall at 85%) was used to compare DANN/DRO/ERM evenly. This is statistically sound for comparing architectures but relies on access to test distributions to calibrate.
- **Missing Variance:** The base-rate sweep lists single FPR values per base rate. Error bars (variance across seeds) are missing for the main base-rate sweep.

**Flag:** $n=5$ for the correlation is a reviewer attack vector. The causal nature (changing *only* the base rate) defends it, but the low $n$ should be explicitly acknowledged.
""")

# 08_COMPUTATIONAL_AUDIT.md
with open(os.path.join(out_dir, "08_COMPUTATIONAL_AUDIT.md"), "w") as f:
    f.write("""# Computational Efficiency Audit

- **Hardware:** 7.8 GB RAM + 4 GB swap.
- **Resolution:** 256x256 (downscaled heavily).
- **Batch Size:** Micro-batch 2 $\times$ accum 4.
- **Architecture:** Dual ResNet-34.

**Verdict:** Calling it "lightweight" is defensible *only relative to the original MVSS-Net (ResNet-50)*. However, a Dual ResNet-34 is objectively heavy for edge deployment. The paper must frame "lightweight" carefully as "resource-constrained trainable" rather than "mobile-ready."
""")

# 09_REPRODUCIBILITY.md
with open(os.path.join(out_dir, "09_REPRODUCIBILITY.md"), "w") as f:
    f.write("""# Reproducibility Audit

**Score: MEDIUM**

- **Pros:** The architectural definitions (`network.py`, `backbone.py`) and experiment logs are highly detailed. The exact hyperparameters (lr=0.0001, epochs=4, pos_weight=100) are recorded in JSON configs.
- **Cons:** The exact dataset loading scripts and preprocessing pipelines for RTM/CASIA are undocumented in the immediate trace. The extreme downsampling (256x256) is a necessary reproducibility detail that might be overlooked by readers expecting standard 512+ resolution for forensics.
""")

# 10_CLAIM_EVIDENCE_MATRIX.md
with open(os.path.join(out_dir, "10_CLAIM_EVIDENCE_MATRIX.md"), "w") as f:
    f.write("""# Claim/Evidence Matrix

| Claim | Evidence in Repo | Status | Recommended Wording |
|---|---|---|---|
| Authentic FPR correlates perfectly with training base rate. | `baserate_experiment_run4.json` yields $\rho=1.0$ across 5 rates. | Directly Supported | "Experiments show a monotonic relationship ($\rho=1.0$) between training forgery prevalence and authentic-image false positive rates." |
| Random initialization separates domains. | Internal logs state ~80% separability. | Directly Supported | "Control experiments using randomly initialized weights demonstrate strong baseline domain separability." |
| Breaking domain-label correlation fixes the generalization. | Decorrelation failed across seeds. | **CONTRADICTED** | *Remove entirely.* Frame as a negative result: structural mitigations fail to resolve the base-rate prior. |
| MVSS-Net-Lite is lightweight. | Dual ResNet-34. | Partially Supported | "MVSS-Net-Lite introduces a reduced-parameter architecture compared to standard ResNet-50 variants..." |
| DANN and DRO fail to mitigate FPR shift. | At 85% recall: DANN (0.36), ERM (0.34), DRO (1.00). | Directly Supported | "Standard domain generalization techniques (DANN, Group-DRO) fail to mitigate the false-positive shift." |
""")

# 11_NOVELTY_AUDIT.md
with open(os.path.join(out_dir, "11_NOVELTY_AUDIT.md"), "w") as f:
    f.write("""# Novelty Audit

A. **Architectural novelty:** Low. Dual-branch ResNets with BayarConv and CBAM are well-established.
B. **Diagnostic/analytical novelty:** **VERY HIGH.** Proving that cross-domain failure is largely a base-rate/prior shift masquerading as feature shift is a massive, highly defensible contribution.
C. **Negative Result novelty:** **HIGH.** Demonstrating that DANN/Group-DRO fail on this specific causal confounder is valuable to the community.

**Risk:** If the paper focuses on the architecture as the main contribution, reviewers will reject it for lack of novelty. The paper *must* pivot entirely around the analytical/diagnostic findings (the base-rate shortcut).
""")

# 12_FAILURES_AND_CONTRADICTIONS.md
with open(os.path.join(out_dir, "12_FAILURES_AND_CONTRADICTIONS.md"), "w") as f:
    f.write("""# Failures and Contradictions

1. **The Decorrelation Fluke:** The initial run at Seed 42 showed a massive win for a feature decorrelation loss. Subsequent runs across different seeds completely eliminated this gain. The repository explicitly acknowledges this ("did NOT replicate"). The paper must not claim this works.
2. **Group DRO Collapse:** Group DRO didn't just fail to improve ERM; it caused a catastrophic collapse (FPR = 1.00 at 85% recall).
3. **Downsampling Confounder:** 256x256 resolution fundamentally destroys pixel-level noise artifacts. The network is forced to learn structural shortcuts because the forensic signal is wiped out. This is a severe failure mode of the hardware constraint, though it elegantly leads to the discovery of the base-rate shortcut.
""")

# 13_PAPER_CODE_CONSISTENCY.md
with open(os.path.join(out_dir, "13_PAPER_CODE_CONSISTENCY.md"), "w") as f:
    f.write("""# Paper ↔ Code Consistency

**LIST A: Fully Supported by Code**
- Dual ResNet-34 architecture with CBAM.
- $\rho=1.0$ monotonic correlation between base rate and FPR.
- The failure of ERM, DANN, and Group DRO (FPR numbers match exact internal logs).
- Random initialization separability (~80%).

**LIST B: Unsupported / Impossible to Verify / Contradicted**
- Any claim that decorrelation or balancing *solves* the issue (Contradicted by multi-seed failure).
- Claims of "state-of-the-art" accuracy (Unsupported; the model was trained under extreme hardware constraints, 256px resolution, and the focus is diagnostic, not SOTA metrics).

**Findings to Add:**
- Explicitly disclose the 256x256 resolution and the 7.8GB RAM constraint as a methodological motivation for the findings.
""")

# 14_FINAL_RESEARCH_AUDIT.md
with open(os.path.join(out_dir, "14_FINAL_RESEARCH_AUDIT.md"), "w") as f:
    f.write("""# Final Research Audit Assessment

1. **Core contribution:** The identification of training base-rate (prevalence) as the dominant causal driver of cross-domain false positives, masquerading as feature shift.
2. **Strongest result:** The base-rate sweep yielding a perfectly monotonic $\rho=1.0$ relationship, cleanly isolating the causal variable.
3. **Weakest result:** The statistical power of the base-rate sweep ($n=5$ points) without explicit variance across multiple seeds for every point.
4. **Biggest methodological weakness:** Severe downscaling to 256x256 due to compute constraints, destroying high-frequency forensic noise.
5. **Biggest reproducibility weakness:** Lack of a frozen, public dataset distribution script for the exact RTM subsets used.
6. **Biggest statistical weakness:** Relying on single runs or discovering a failure only after multi-seed replication (the seed-42 fluke).
7. **Biggest novelty risk:** Positioning the paper as an "Architecture" paper rather than a "Diagnostic/Analysis" paper.
8. **Biggest reviewer attack vector:** "You only tested on 256x256 resolution, of course your model learned a dataset-level prior instead of forensic noise."
9. **Most important missing experiment:** Testing if a dynamically calibrated, domain-specific decision threshold (e.g., using a validation set from the target domain) mathematically zeroes out the FPR shift.
10. **Claims that must be weakened:** "MVSS-Net-Lite solves domain generalization" $\rightarrow$ "MVSS-Net-Lite exposes the causal limits of domain generalization."
11. **Claims that are safe:** The diagnostic findings, the failure of DANN/DRO, and the $\rho=1.0$ correlation.
12. **Experiments that should be added:** A sweep at 512x512 (if compute allows) to see if the base-rate reliance vanishes when true forensic noise is preserved.
13. **Figures/tables that should be added:** A clear table comparing Seed 42 (fluke) vs Seed X (failure) for the decorrelation loss, showing intellectual honesty.
14. **Things that should be removed:** Any optimistic language about decorrelation fixing the issue.
15. **Overall research maturity:** **PAPER-READY WITH REVISIONS.** The negative results and diagnostic insights are highly publishable, provided the architectural claims are strictly reigned in.
""")

# MASTER_RESULTS.json
master_results = [
    {
        "experiment_id": "EXP-01",
        "metric": "Spearman Rho",
        "value": 1.0,
        "unit": "correlation",
        "dataset": "RTM",
        "model": "MVSS-Net-Lite",
        "split": "Test",
        "seed": 42,
        "source_file": "reports/baserate_experiment_run4.json",
        "source_line_or_artifact": "Base rate sweep {0, .25, .5, .75, 1.0}",
        "reproducible": True,
        "notes": "Perfect monotonic correlation between base rate and authentic FPR."
    },
    {
        "experiment_id": "EXP-01",
        "metric": "Pearson r",
        "value": 0.84,
        "unit": "correlation",
        "dataset": "RTM",
        "model": "MVSS-Net-Lite",
        "split": "Test",
        "seed": 42,
        "source_file": "paper1_plan.md",
        "source_line_or_artifact": "r=0.84",
        "reproducible": True,
        "notes": "Linear correlation for the base rate sweep."
    },
    {
        "experiment_id": "EXP-03",
        "metric": "FPR at 85% Recall",
        "value": 0.34,
        "unit": "rate",
        "dataset": "RTM",
        "model": "ERM Baseline",
        "split": "Cross-Domain",
        "seed": "UNKNOWN",
        "source_file": "paper1_plan.md",
        "source_line_or_artifact": "ERM FPR approx 0.34",
        "reproducible": True,
