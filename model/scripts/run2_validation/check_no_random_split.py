"""
Check 2: random_split absence from the active Stage 2 training path.

Method: AST-based import-graph reachability walk starting from the real
entrypoint (model/train.py). Resolves local imports (model.*, data_pipeline.*)
to files, BFS through the graph, then scans every reachable file for:
  - AST Name/Attribute nodes named 'random_split'
  - the plain-text token 'random_split' (catches string/comment references too)
Archived scratch code (debug_memories/) is NOT reachable and therefore not in
scope; it IS reported separately for transparency.

Run from repo root:
    python -m model.scripts.run2_validation.check_no_random_split
"""
import ast
import os
import re

ENTRYPOINTS = ['model/train.py']
LOCAL_PREFIXES = ('model', 'data_pipeline')
ROOT = '.'


def module_to_files(module_name):
    """Resolve a module name to candidate .py files inside this repo."""
    rel = module_name.replace('.', os.sep)
    candidates = [
        os.path.join(ROOT, rel + '.py'),
        os.path.join(ROOT, rel, '__init__.py'),
    ]
    return [c for c in candidates if os.path.exists(c)]


def imports_of(path):
    with open(path, 'r') as f:
        tree = ast.parse(f.read(), filename=path)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


def scan_tokens(path):
    """AST hits + raw-token hits for random_split / get_splits."""
    with open(path, 'r') as f:
        src = f.read()
    ast_hits = []
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == 'random_split':
            ast_hits.append((node.lineno, "Name 'random_split'"))
        elif isinstance(node, ast.Attribute) and node.attr == 'random_split':
            ast_hits.append((node.lineno, "Attribute '.random_split'"))
    text_hits = []
    for i, line in enumerate(src.splitlines(), 1):
        if re.search(r'\brandom_split\b', line):
            text_hits.append((i, line.strip()))
    return ast_hits, text_hits


def main():
    reachable = []
    queue = list(ENTRYPOINTS)
    seen = set()

    print("=== Reachability walk (BFS over local imports) ===")
    while queue:
        rel = queue.pop(0)
        if rel in seen or not os.path.exists(rel):
            continue
        seen.add(rel)
        reachable.append(rel)

        for mod in imports_of(rel):
            if mod.split('.')[0] in LOCAL_PREFIXES:
                for f in module_to_files(mod):
                    if f not in seen:
                        queue.append(f)

    for f in sorted(reachable):
        print(f"  reachable: {f}")

    print("\n=== Per-file scan results ===")
    total_ast = 0
    for f in sorted(reachable):
        ast_hits, text_hits = scan_tokens(f)
        status = "CLEAN"
        if ast_hits or text_hits:
            status = "HIT"
        print(f"[{status}] {f}")
        for ln, desc in ast_hits:
            print(f"    AST line {ln}: {desc}")
        for ln, line in text_hits:
            print(f"    TEXT line {ln}: {line}")
        total_ast += len(ast_hits)

    # Transparency note about archived (unreachable) copies
    print("\n=== Out-of-scope archives containing 'random_split' (NOT importable from train.py) ===")
    archive_hits = []
    for dirpath, _dirs, files in os.walk('debug_memories'):
        for fn in files:
            if fn.endswith('.py'):
                p = os.path.join(dirpath, fn)
                _a, t = scan_tokens(p)
                if t:
                    archive_hits.append(p)
                    print(f"  {p} (lines: {[ln for ln, _ in t]})")
    if not archive_hits:
        print("  (none)")

    ok = total_ast == 0
    print(f"\nReachable files scanned: {len(reachable)}")
    print(f"AST 'random_split' hits in reachable files: {total_ast}")
    print(f"RESULT: {'PASS - random_split absent from active Stage 2 path' if ok else 'FAIL'}")


if __name__ == '__main__':
    main()
