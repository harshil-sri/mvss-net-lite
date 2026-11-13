import re

with open('/home/harshil/.gemini/antigravity-cli/brain/c5e2e37d-2ff8-4c4a-b29e-552f13f1c0d6/.system_generated/tasks/task-730.log', 'r') as f:
    log_content = f.read()

results = {}
current_chkpt = None
for line in log_content.split('\n'):
    if line.startswith('Evaluating:'):
        current_chkpt = line.split('_ep')[1].split('.pt')[0]
        if current_chkpt == "10_bestval":
             current_chkpt = "10 (Best)"
        results[current_chkpt] = []
    elif 'FP' in line and current_chkpt:
        parts = line.split(':')
        val = parts[1].strip()
        results[current_chkpt].append(val)

probe_table = ""
for chkpt, vals in results.items():
    if len(vals) == 4:
        probe_table += f"| Epoch {chkpt} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |\n"

with open('reports/diagnostic_stage2run3_report.md', 'r') as f:
    content = f.read()

header = "| Checkpoint | RTM Auth FP (Thresh 0.5) | RTM Auth FP (Thresh 0.9) | MIDV500 Auth FP (Thresh 0.5) | MIDV500 Auth FP (Thresh 0.9) |\n|---|---|---|---|---|\n"
replacement = header + probe_table

content = content.replace("*(Waiting for probe results...)*", replacement)

with open('reports/diagnostic_stage2run3_report.md', 'w') as f:
    f.write(content)

with open('/home/harshil/.gemini/antigravity-cli/brain/c5e2e37d-2ff8-4c4a-b29e-552f13f1c0d6/diagnostic_stage2run3_report.md', 'w') as f:
    f.write(content)
