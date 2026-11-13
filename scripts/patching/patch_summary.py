import re
with open('model/train.py', 'r') as f:
    content = f.read()

search = """        print(f"=== Epoch {epoch} Summary ===")
        print(f"Time: {epoch_duration:.1f}s | LR: {current_lr:.6f} | GPU Mem: {gpu_mem:.0f} MB")
        print(f"TRAIN -> Avg Seg: {avg_seg:.4f} | Avg Edge: {avg_edge:.4f} | Avg Total: {avg_total:.4f}")
        print(f"VAL   -> Avg Seg: {avg_val_seg:.4f} | Avg Edge: {avg_val_edge:.4f} | Avg Total: {avg_val_total:.4f}\\n")"""

replace = """
        domain_acc = epoch_domain_correct / max(1, epoch_domain_total) if epoch_domain_total > 0 else 0.0
        if scheduler is not None:
            scheduler.step()

        print(f"=== Epoch {epoch} Summary ===")
        print(f"Time: {epoch_duration:.1f}s | LR: {current_lr:.6f} | GPU Mem: {gpu_mem:.0f} MB")
        if args.domain_adversarial:
            print(f"TRAIN -> Avg Seg: {avg_seg:.4f} | Avg Edge: {avg_edge:.4f} | Avg Total: {avg_total:.4f} | Domain Acc: {domain_acc*100:.2f}%")
        else:
            print(f"TRAIN -> Avg Seg: {avg_seg:.4f} | Avg Edge: {avg_edge:.4f} | Avg Total: {avg_total:.4f}")
        print(f"VAL   -> Avg Seg: {avg_val_seg:.4f} | Avg Edge: {avg_val_edge:.4f} | Avg Total: {avg_val_total:.4f}\\n")
"""

content = content.replace(search, replace)
with open('model/train.py', 'w') as f:
    f.write(content)
