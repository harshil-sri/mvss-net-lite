import re
with open('model/train.py', 'r') as f:
    content = f.read()

# Replace rotate_checkpoints
search = r"def rotate_checkpoints.*?return removed"
replace = """def rotate_checkpoints(saved_checkpoints, keep=3):
    \"\"\"
    Keeps only the most recent `keep` checkpoints. Mutates saved_checkpoints in place.
    Returns the list of paths actually deleted.
    \"\"\"
    if len(saved_checkpoints) <= keep:
        return []

    removed = []
    candidates = saved_checkpoints[:-keep].copy()
    for c in candidates:
        if os.path.exists(c):
            os.remove(c)
        removed.append(c)
        saved_checkpoints.remove(c)

    return removed"""

content = re.sub(search, replace, content, flags=re.DOTALL)

with open('model/train.py', 'w') as f:
    f.write(content)
