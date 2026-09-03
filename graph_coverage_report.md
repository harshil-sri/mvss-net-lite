# Graph Coverage Report

## Summary
- **Total files in repository (approx):** 260,572
- **Total files included in graph (approx):** 450

## Excluded Paths & Reasons
To provide a focused, navigable structure for Claude Opus 5 without overwhelming the context with massive binary files, virtual environments, or noise, the following patterns were excluded via `.graphifyignore` (based on actual measured disk usage):

1. `venv/`, `node_modules/`, `__pycache__/`, `.git/` - Standard noise, caching, and internal history lacking immediate source navigation value.
2. `model/checkpoints/` and `mock_chkpts/` and `*.pth`/`*.pt` - 13.75 GB of binary model weights (no AST/navigation value).
3. `data/` and `data_pipeline/raw/` - 76.43 GB containing ~259,000 raw dataset image files (RTM/MIDV500/CASIAv2/DEFACTO) which adds overwhelming scale without structural value.
4. `debug_memories/` - 6.63 MB archive of previous debugging sessions and heatmaps (adds size without core structural codebase value).
5. `backend/app/static/uploads/` - Arbitrary user-uploaded images dumped during frontend testing (pure noise).
6. `reports/manifest*.json` and `graphify-out/manifest.json` - The full JSON manifests were excluded to prevent the graph from ingesting millions of JSON lines. (Their schema has been included instead via `manifest_schema.json` and `manifest_v2_schema.json`).

## Known Missing Files (Flagged)
The following files were named in earlier debugging reports but have been permanently deleted from disk. They have been explicitly flagged as **MISSING** in the graph structure (via `MISSING_FILES_AUDIT.md`) so the agent is aware of their absence:
- `validate_midv500_synthetic.py`
- `reconciliation_probe.py`

*(Note: `probe_blur.py` and `generate_midv500_synthetic.py` were recovered and are correctly mapped in the graph under `diagnostics/`. `dataset_loader.py` is present and mapped under `data_pipeline/`).*
