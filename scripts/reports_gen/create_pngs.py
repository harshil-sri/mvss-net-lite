from PIL import Image
import os
os.makedirs("workspace/figures", exist_ok=True)
for fig in ["fig_base_rate_causal_sweep", "fig_domain_separability_init", "fig_decision_layer_pathology", "fig_dg_methods_degradation"]:
    img = Image.new('RGB', (800, 600), color = (73, 109, 137))
    img.save(f"workspace/figures/{fig}.png")
