import re

# Update REQUIRES_HUMAN_REVIEW.md
with open("REQUIRES_HUMAN_REVIEW.md", "r") as f:
    text = f.read()

# Only keep items that are still unverified/require review
text = re.sub(r'## H1.*?---', '', text, flags=re.DOTALL) # H1 resolved
text = re.sub(r'## H3.*?---', '', text, flags=re.DOTALL) # H3 resolved
text = re.sub(r'## H7.*?---', '', text, flags=re.DOTALL) # H7 resolved

with open("REQUIRES_HUMAN_REVIEW.md", "w") as f:
    f.write(text)


# Update RECONCILIATION_REPORT.md
with open("RECONCILIATION_REPORT.md", "r") as f:
    text = f.read()

text += """
## Post-Orchestrator Evidence Resolutions

| Item | Resolution | Evidence Found |
|---|---|---|
| **H1** (LODO Head) | **CORRECTED** | Raw detection head AUCs (0.6567–0.6794) and ERM (0.7139) support an envelope of `0.65 to 0.68` and `−0.034 to −0.057`. The unsupported `-0.064` was corrected to `-0.057`. |
| **H2** (DG Pilot) | **UNVERIFIED** | The exact metrics (FP @ 85% recall) for DANN 0.36 and Group-DRO 1.00 were not found in the raw pilot logs, which evaluate at threshold 0.5. |
| **H3** (p-value) | **CONFIRMED** | `reports/pathb_baserate_experiment.json:60` contains `"spearman_p": 0.0374`. |
| **H4** (Bayar probe) | **UNVERIFIED** | `+0.2992` found, but the raw log for `+0.1087` was overwritten during development. |
| **H7** (Root causes)| **REWORDED** | `mvss_net_debugging_summary (3).md` lists 21 items, but includes non-defects (e.g., generating a codebase graph). "root causes" changed to "development issues", and "five thematic groups" corrected to "six". |
"""
with open("RECONCILIATION_REPORT.md", "w") as f:
    f.write(text)
