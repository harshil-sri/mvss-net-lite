import re
with open('model/train.py', 'r') as f:
    train_code = f.read()

# 1. Add args for lr-schedule, early-stopping-patience, domain-adversarial
args_block = """
    parser.add_argument('--use-balanced-sampler', action='store_true', help='Use balanced sampler for target classes')
    parser.add_argument('--lr-schedule', type=str, default='flat', choices=['flat', 'cosine'], help='Learning rate schedule')
    parser.add_argument('--early-stopping-patience', type=int, default=-1, help='Epochs to wait before early stop (-1 to disable)')
    parser.add_argument('--domain-adversarial', action='store_true', help='Use domain adversarial loss')
"""
train_code = re.sub(r"parser\.add_argument\('--use-balanced-sampler'[^\n]+", args_block.strip(), train_code)

# 2. Add cosine scheduler and best val logic, and DANN lambda computation
# Wait, I will just rewrite model/train.py piece by piece.
