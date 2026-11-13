import re
import sys

with open('model/train.py', 'r') as f:
    content = f.read()

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
            
            domain_loss_t = 0.0
            domain_correct = 0
            domain_total = 0
            if args.domain_adversarial and sampler_classes is not None:
                labels_str = [sampler_classes[idx] for idx in sample_idxs.tolist()]
                valid_mask = torch.tensor([l in ('rtm_forged', 'rtm_auth', 'midv500') for l in labels_str], device=device)
                if valid_mask.sum() > 0:
                    domain_targets = torch.tensor([1 if l == 'midv500' else 0 for l in labels_str], device=device)
                    domain_logits_valid = domain_logits[valid_mask]
                    domain_targets_valid = domain_targets[valid_mask]
                    
                    domain_loss_t = F.cross_entropy(domain_logits_valid, domain_targets_valid)
                    
                    domain_preds = torch.argmax(domain_logits_valid, dim=1)
                    domain_correct += (domain_preds == domain_targets_valid).sum().item()
                    domain_total += len(domain_targets_valid)

            loss_total = (SEG_LOSS_WEIGHT * loss_seg) + (EDGE_LOSS_WEIGHT * loss_edge)
            if args.domain_adversarial:
                loss_total += 0.1 * domain_loss_t
"""
content = re.sub(train_batch_search, train_batch_replace, content)

# 5. Training loop Accumulations
train_acc_search = r"            # Accumulate\n            epoch_seg_loss \+= loss_seg.item\(\)\n            epoch_edge_loss \+= loss_edge.item\(\)\n            epoch_total_loss \+= loss_total.item\(\)\n            batches_processed \+= 1\n"
train_acc_replace = """            # Accumulate
            epoch_seg_loss += loss_seg.item()
            epoch_edge_loss += loss_edge.item()
            epoch_total_loss += loss_total.item()
            batches_processed += 1
            global_step += 1
            if args.domain_adversarial and sampler_classes is not None and domain_total > 0:
                epoch_domain_correct += domain_correct
                epoch_domain_total += domain_total
"""
content = re.sub(train_acc_search, train_acc_replace, content)

# 8. Val Loop forwarding
val_batch_search = r"                v_pred_seg, v_pred_edge = model\(v_imgs\)\n"
val_batch_replace = """                if args.domain_adversarial:
                    v_pred_seg, v_pred_edge, _ = model(v_imgs, lambda_p=0.0)
                else:
                    v_pred_seg, v_pred_edge = model(v_imgs)\n"""
content = re.sub(val_batch_search, val_batch_replace, content)

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
        if epoch % SAVE_EVERY == 0 or epoch == total_epochs:\n"""
content = re.sub(ckpt_search, ckpt_replace, content)

# Delete old bestval rotation logic
del_bestval_search = r"            if avg_val_total < best_val_loss:\n                best_val_loss = avg_val_total\n                best_checkpoint_path = chkpt_path\n                print\(f\"New best validation loss: \{best_val_loss:\.4f\} at epoch \{epoch\}\"\)\n"
content = re.sub(del_bestval_search, "", content)

# 11. Halt on ES
es_halt_search = r"    # Plotting the loss curve\n"
es_halt_replace = """        if es_stop:
            print(f"Early stopping triggered at epoch {epoch}. Halting training.")
            break

    # Plotting the loss curve\n"""
content = re.sub(es_halt_search, es_halt_replace, content)

with open('model/train.py', 'w') as f:
    f.write(content)

print("Patching done!")
