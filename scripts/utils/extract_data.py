import csv

# Task 1: Fig 4 Data
with open("handoff/fig4_base_rate_sweep.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "train_base_rate", "authentic_fp_at_85_recall", "seed"])
    
    # MVSS
    writer.writerow(["MVSS-Net-Lite", "0.0", "0.0", "42"])
    writer.writerow(["MVSS-Net-Lite", "0.25", "0.5858", "42"])
    writer.writerow(["MVSS-Net-Lite", "0.5", "0.589", "42"])
    writer.writerow(["MVSS-Net-Lite", "0.75", "0.6699", "42"])
    writer.writerow(["MVSS-Net-Lite", "1.0", "0.7476", "42"])
    
    # ResNet-UNet/FPN
    writer.writerow(["ResNet-UNet/FPN", "0.0", "0.1683", "42"])
    writer.writerow(["ResNet-UNet/FPN", "0.25", "0.7961", "42"])
    writer.writerow(["ResNet-UNet/FPN", "0.5", "0.7314", "42"])
    writer.writerow(["ResNet-UNet/FPN", "0.75", "0.9773", "42"])
    writer.writerow(["ResNet-UNet/FPN", "1.0", "1.0", "42"])

# Task 2: Fig 2 Data
DOMAIN_ACC = [94.07, 96.09, 94.63, 85.45, 85.82, 75.82, 68.20, 68.84, 68.04,
              69.55, 71.76, 67.10, 67.27, 67.08, 64.40, 65.38, 66.54, 66.50,
              66.51, 64.65, 64.72, 65.96, 63.93, 65.65, 63.60, 63.63, 64.92,
              62.50]

with open("reports/stage2run4_history.csv", "r") as f_in, open("handoff/fig2_run4_history.csv", "w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)
    writer.writerow(["epoch", "train_total_loss", "val_total_loss", "domain_accuracy_percent"])
    for i, row in enumerate(reader):
        writer.writerow([row["epoch"], row["train_total_loss"], row["val_total_loss"], DOMAIN_ACC[i]])

# Task 3: Fig 3 Data
with open("handoff/fig3_dg.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "auth_fp_at_85_recall"])
    writer.writerow(["image-level detection head", "0.29"])
    writer.writerow(["ERM", "0.34"])
    writer.writerow(["DANN", "0.36"])
    writer.writerow(["Group-DRO", "1.00"])
