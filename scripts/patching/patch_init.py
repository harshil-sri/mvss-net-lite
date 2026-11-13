import re
with open('model/train.py', 'r') as f:
    content = f.read()

search = """        epoch_seg_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_total_loss = 0.0
        batches_processed = 0"""

replace = """        epoch_seg_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_total_loss = 0.0
        batches_processed = 0
        epoch_domain_correct = 0
        epoch_domain_total = 0"""

content = content.replace(search, replace)
with open('model/train.py', 'w') as f:
    f.write(content)
