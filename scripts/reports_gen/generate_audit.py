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
