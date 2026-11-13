import re
with open('model/train.py', 'r') as f:
    content = f.read()

args_search = r"    parser.add_argument\(\"--use-balanced-sampler\".*?\)\n"
args_replace = """    parser.add_argument("--use-balanced-sampler", action='store_true', help="Enable WeightedRandomSampler for dataset balancing")
    parser.add_argument('--lr-schedule', type=str, default='flat', choices=['flat', 'cosine'], help='Learning rate schedule')
    parser.add_argument('--early-stopping-patience', type=int, default=-1, help='Epochs to wait before early stop (-1 to disable)')
    parser.add_argument('--domain-adversarial', action='store_true', help='Use domain adversarial loss')\n"""
content = re.sub(args_search, args_replace, content)
with open('model/train.py', 'w') as f:
    f.write(content)
