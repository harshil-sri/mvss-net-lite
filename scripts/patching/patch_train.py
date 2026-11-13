import re
import sys

with open('model/train.py', 'r') as f:
    content = f.read()

# 1. Parse Args
args_search = r"parser\.add_argument\('--use-balanced-sampler'.*?\n"
args_replace = """    parser.add_argument('--use-balanced-sampler', action='store_true', help='Use balanced sampler for target classes')
    parser.add_argument('--lr-schedule', type=str, default='flat', choices=['flat', 'cosine'], help='Learning rate schedule')
    parser.add_argument('--early-stopping-patience', type=int, default=-1, help='Epochs to wait before early stop (-1 to disable)')
    parser.add_argument('--domain-adversarial', action='store_true', help='Use domain adversarial loss')\n"""
content = re.sub(args_search, args_replace, content)

# 2. Imports and Initializations
init_search = r"import sys\n"
init_replace = "import sys\nimport math\nimport torch.nn.functional as F\n"
content = content.replace(init_search, init_replace)

# 3. Before training loop: LR Scheduler, Domain criterion, DANN params
pre_loop_search = r"    halted_low_disk = False\n\n    for epoch in range\(start_epoch, total_epochs \+ 1\):\n"
pre_loop_replace = """    halted_low_disk = False
    
    # Setup LR Scheduler
    scheduler = None
    if args.lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    # Setup early stopping
    es_patience = args.early_stopping_patience
    es_counter = 0
    es_best_loss = float('inf')
    es_stop = False
    
    # Setup best val logic
    global_best_val_loss = float('inf')
    best_val_checkpoint_path = None
    
    # Pre-compute total batches for lambda schedule
    total_steps = len(train_loader) * total_epochs
    global_step = 0
    
    for epoch in range(start_epoch, total_epochs + 1):
"""
content = content.replace(pre_loop_search, pre_loop_replace)

# 4. Training loop: domain targets, domain loss
train_batch_search = r"            # Forward pass\n            # MVSSNetLite forward returns \(seg_mask, edge_map\)\n            pred_seg, pred_edge = model\(imgs\)\n\n            # Compute losses\n            loss_seg = seg_criterion\(pred_seg, masks\)\n            loss_edge = edge_criterion\(pred_edge, edges\)\n\n            loss_total = \(SEG_LOSS_WEIGHT \* loss_seg\) \+ \(EDGE_LOSS_WEIGHT \* loss_edge\)\n"
train_batch_replace = """            # GRL lambda calculation
            lambda_p = None
            if args.domain_adversarial:
                p = global_step / float(total_steps)
                lambda_p = (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0

            # Forward pass
            if args.domain_adversarial:
                pred_seg, pred_edge, domain_logits = model(imgs, lambda_p=lambda_p)
            else:
                pred_seg, pred_edge = model(imgs)

            # Compute losses
            loss_seg = seg_criterion(pred_seg, masks)
            loss_edge = edge_criterion(pred_edge, edges)
            
            domain_loss = 0.0
            domain_correct = 0
            domain_total = 0
            if args.domain_adversarial and sampler_classes is not None:
                # Map sample_idxs to domains: RTM=0, MIDV500=1
                # 'rtm_forged' and 'rtm_auth' -> 0, 'midv500' -> 1
                labels_str = [sampler_classes[idx] for idx in sample_idxs.tolist()]
                # filter out 'other'
                valid_mask = torch.tensor([l in ('rtm_forged', 'rtm_auth', 'midv500') for l in labels_str], device=device)
                if valid_mask.sum() > 0:
                    domain_targets = torch.tensor([1 if l == 'midv500' else 0 for l in labels_str], device=device)
                    # Only calculate loss on valid samples
                    domain_logits_valid = domain_logits[valid_mask]
                    domain_targets_valid = domain_targets[valid_mask]
                    
                    domain_loss_t = F.cross_entropy(domain_logits_valid, domain_targets_valid)
                    domain_loss = domain_loss_t.item()
                    
                    domain_preds = torch.argmax(domain_logits_valid, dim=1)
                    domain_correct += (domain_preds == domain_targets_valid).sum().item()
                    domain_total += len(domain_targets_valid)
                else:
                    domain_loss_t = 0.0
            else:
                domain_loss_t = 0.0

            loss_total = (SEG_LOSS_WEIGHT * loss_seg) + (EDGE_LOSS_WEIGHT * loss_edge)
            if args.domain_adversarial:
                loss_total += 0.1 * domain_loss_t
"""
content = content.replace(train_batch_search, train_batch_replace)

# 5. Training loop Accumulations
train_acc_search = r"            # Accumulate\n            epoch_seg_loss \+= loss_seg.item\(\)\n            epoch_edge_loss \+= loss_edge.item\(\)\n            epoch_total_loss \+= loss_total.item\(\)\n            batches_processed \+= 1\n"
train_acc_replace = """            # Accumulate
            epoch_seg_loss += loss_seg.item()
            epoch_edge_loss += loss_edge.item()
            epoch_total_loss += loss_total.item()
            batches_processed += 1
            global_step += 1
"""
content = content.replace(train_acc_search, train_acc_replace)

# 6. Init missing accumulators at epoch start
epoch_start_search = r"        epoch_seg_loss, epoch_edge_loss, epoch_total_loss = 0.0, 0.0, 0.0\n        batches_processed = 0\n"
epoch_start_replace = """        epoch_seg_loss, epoch_edge_loss, epoch_total_loss = 0.0, 0.0, 0.0
        batches_processed = 0
        epoch_domain_correct = 0
        epoch_domain_total = 0
"""
content = content.replace(epoch_start_search, epoch_start_replace)

# 7. Add domain_correct / total in batch loop
train_batch_acc_search = r"            epoch_total_loss \+= loss_total.item\(\)\n            batches_processed \+= 1\n            global_step \+= 1\n"
train_batch_acc_replace = """            epoch_total_loss += loss_total.item()
            batches_processed += 1
            global_step += 1
            if args.domain_adversarial and sampler_classes is not None and domain_total > 0:
                epoch_domain_correct += domain_correct
                epoch_domain_total += domain_total
"""
content = content.replace(train_batch_acc_search, train_batch_acc_replace)

# 8. Val Loop forwarding
val_batch_search = r"                v_pred_seg, v_pred_edge = model\(v_imgs\)\n"
val_batch_replace = """                if args.domain_adversarial:
                    v_pred_seg, v_pred_edge, _ = model(v_imgs, lambda_p=0.0) # lambda=0 for eval to just get logits if needed
                else:
                    v_pred_seg, v_pred_edge = model(v_imgs)
"""
content = content.replace(val_batch_search, val_batch_replace)

# 9. Epoch Summary and Scheduler step
epoch_summary_search = r"        print\(f\"=== Epoch \{epoch\} Summary ===\"\)\n        print\(f\"Time: \{epoch_duration:\.1f\}s \| LR: \{current_lr:\.6f\} \| GPU Mem: \{gpu_mem:\.0f\} MB\"\)\n        print\(f\"TRAIN -> Avg Seg: \{avg_seg:\.4f\} \| Avg Edge: \{avg_edge:\.4f\} \| Avg Total: \{avg_total:\.4f\}\"\)\n        print\(f\"VAL   -> Avg Seg: \{avg_val_seg:\.4f\} \| Avg Edge: \{avg_val_edge:\.4f\} \| Avg Total: \{avg_val_total:\.4f\}\\n\"\)\n"
epoch_summary_replace = """
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
content = content.replace(epoch_summary_search, epoch_summary_replace)

# 10. Bestval Checkpointing and Early Stopping
ckpt_search = r"        # Save checkpoint \(selectively, with rotation \+ disk-space guardrail\)\n        if epoch % SAVE_EVERY == 0 or epoch == total_epochs:\n"
ckpt_replace = """        # Check best validation loss and save _bestval checkpoint
        if avg_val_total < global_best_val_loss:
            global_best_val_loss = avg_val_total
            bestval_path = f"model/checkpoints/{args.stage_name}_mvss_lite_ep{epoch}_bestval.pt"
            
            payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_total
            }
            torch.save(payload, bestval_path)
            print(f"*** New best validation total loss: {avg_val_total:.4f} at epoch {epoch}. Saved {bestval_path} ***")
            
            # Remove the previous bestval checkpoint if it exists and is different
            if best_val_checkpoint_path and best_val_checkpoint_path != bestval_path and os.path.exists(best_val_checkpoint_path):
                os.remove(best_val_checkpoint_path)
            best_val_checkpoint_path = bestval_path
            
            es_counter = 0
        else:
            es_counter += 1
            print(f"Early stopping counter: {es_counter} out of {es_patience}")
            
        if es_patience > 0 and es_counter >= es_patience:
            es_stop = True

        # Save checkpoint (selectively, with rotation + disk-space guardrail)
        if epoch % SAVE_EVERY == 0 or epoch == total_epochs:
"""
content = content.replace(ckpt_search, ckpt_replace)

# Delete old bestval rotation logic
del_bestval_search = r"            if avg_val_total < best_val_loss:\n                best_val_loss = avg_val_total\n                best_checkpoint_path = chkpt_path\n                print\(f\"New best validation loss: \{best_val_loss:\.4f\} at epoch \{epoch\}\"\)\n"
content = re.sub(del_bestval_search, "", content)

# Modify rotation function signature and call
rot_call_search = r"            removed = rotate_checkpoints\(saved_checkpoints, best_checkpoint_path\)\n"
rot_call_replace = """            # Do not pass best_checkpoint_path to rotation, it's handled separately
            removed = rotate_checkpoints(saved_checkpoints)
"""
content = re.sub(rot_call_search, rot_call_replace, content)

rot_def_search = r"def rotate_checkpoints\(checkpoints_list, best_checkpoint_path, keep_last=3\):\n    \"\"\"\n    Keeps the last N checkpoints \+ the all-time best checkpoint.\n    Returns a list of removed file paths.\n    \"\"\"\n    if len\(checkpoints_list\) <= keep_last:\n        return \[\]\n\n    removed = \[\]\n    # We only want to evaluate the ones older than the 'keep_last' window\n    candidates_for_removal = checkpoints_list\[:-keep_last\]\n    for c in candidates_for_removal:\n        if c != best_checkpoint_path:\n            if os\.path\.exists\(c\):\n                os\.remove\(c\)\n            removed\.append\(c\)\n            checkpoints_list\.remove\(c\)\n    return removed\n"
rot_def_replace = """def rotate_checkpoints(checkpoints_list, keep_last=3):
    \"\"\"
    Keeps the last N checkpoints.
    Returns a list of removed file paths.
    \"\"\"
    if len(checkpoints_list) <= keep_last:
        return []

    removed = []
    candidates_for_removal = checkpoints_list[:-keep_last].copy()
    for c in candidates_for_removal:
        if os.path.exists(c):
            os.remove(c)
        removed.append(c)
        checkpoints_list.remove(c)
    return removed
"""
content = re.sub(rot_def_search, rot_def_replace, content)

# 11. Halt on ES
es_halt_search = r"    # Plotting the loss curve\n"
es_halt_replace = """        if es_stop:
            print(f"Early stopping triggered at epoch {epoch}. Halting training.")
            break

    # Plotting the loss curve\n"""
content = content.replace(es_halt_search, es_halt_replace)

# 12. Fix the initial best_val_loss and best_checkpoint_path assignment in original code
rem_init_bestval = r"    best_val_loss = float\('inf'\)\n    best_checkpoint_path = None\n"
content = re.sub(rem_init_bestval, "", content)


with open('model/train_new.py', 'w') as f:
    f.write(content)

print("Patching done")
