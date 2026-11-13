import re
with open('model/train.py', 'r') as f:
    content = f.read()

search = """def rotate_checkpoints(saved_checkpoints, best_checkpoint_path, keep=3):
    \"\"\"
    Keeps only the most recent `keep` checkpoints, never deleting the current
    best-validation checkpoint. Mutates saved_checkpoints in place.
    Returns the list of paths actually deleted.
    \"\"\"
    if len(saved_checkpoints) <= keep:
        return []

    removed = []
    # All checkpoints older than the keep window are candidates for deletion
    candidates = saved_checkpoints[:-keep]
    for c in candidates:
        if c != best_checkpoint_path:
            if os.path.exists(c):
                os.remove(c)
            removed.append(c)
            saved_checkpoints.remove(c)

    return removed"""

replace = """def rotate_checkpoints(saved_checkpoints, keep=3):
    \"\"\"
    Keeps only the most recent `keep` checkpoints. Mutates saved_checkpoints in place.
    Returns the list of paths actually deleted.
    \"\"\"
    if len(saved_checkpoints) <= keep:
        return []

    removed = []
    # All checkpoints older than the keep window are candidates for deletion
    candidates = saved_checkpoints[:-keep].copy()
    for c in candidates:
        if os.path.exists(c):
            os.remove(c)
        removed.append(c)
        saved_checkpoints.remove(c)

    return removed"""

content = content.replace(search, replace)
with open('model/train.py', 'w') as f:
    f.write(content)
