import re
with open('model/train.py', 'r') as f:
    content = f.read()

search = r"    halted_low_disk = False\n\n    for epoch in range\(start_epoch, total_epochs \+ 1\):\n"
replace = """    halted_low_disk = False
    
    # Setup LR Scheduler
    scheduler = None
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
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

content = re.sub(search, replace, content)
with open('model/train.py', 'w') as f:
    f.write(content)
