import re
with open('model/train.py', 'r') as f:
    content = f.read()

search = "import sys\n"
replace = "import sys\nimport math\nimport torch.nn.functional as F\n"
content = content.replace(search, replace)

with open('model/train.py', 'w') as f:
    f.write(content)
