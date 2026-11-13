import re

with open('diagnostics/probe_blur_v3.py', 'r') as f:
    code = f.read()

# 1. Remove the deliberate crash
deliberate_crash = """            # DELIBERATE SMOKE TEST SKIP
            if idx_in_list == 0:
                raise RuntimeError("Deliberate smoke test skip triggered")
"""
code = code.replace(deliberate_crash, "")

# 2. Add argparse import and __main__ block replacement
if "import argparse" not in code:
    code = code.replace("import os\n", "import os\nimport argparse\n")

main_block = """if __name__ == '__main__':
    run_probe(num_samples=7, smoketest=True)"""

new_main_block = """if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run probe blur audit")
    parser.add_argument('--num-samples', type=int, default=200, help="Target number of samples per setting")
    parser.add_argument('--smoketest', action='store_true', help="Run in smoketest mode")
    args = parser.parse_args()
    
    run_probe(num_samples=args.num_samples, smoketest=args.smoketest)"""
code = code.replace(main_block, new_main_block)

with open('diagnostics/probe_blur_v3.py', 'w') as f:
    f.write(code)
