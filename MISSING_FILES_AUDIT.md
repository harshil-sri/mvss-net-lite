# Missing Files Audit

The following critical scripts are explicitly flagged as **MISSING** because they were deleted from the disk during a prior cleanup routine and no longer exist in the repository:

- `validate_midv500_synthetic.py`
- `reconciliation_probe.py`

*(Note: `probe_blur.py` and `generate_midv500_synthetic.py` were previously missing but have since been recovered to the `diagnostics/` folder. `dataset_loader.py` is present in `data_pipeline/`)*.

This document serves to inject these known missing paths into the graphify structure so that future agents are aware of their absence and do not attempt to read them or assume they were silently omitted by the graph extraction process.
