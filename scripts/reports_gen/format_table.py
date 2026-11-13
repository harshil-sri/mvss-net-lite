with open('reports/tmux_stage2run2.log', 'r') as f:
    lines = f.readlines()

trains = []
vals = []

for line in lines:
    if "TRAIN -> Avg Seg:" in line:
        parts = line.split('|')
        s = parts[0].split(':')[1].strip()
        e = parts[1].split(':')[1].strip()
        t = parts[2].split(':')[1].strip()
        trains.append((s, e, t))
    elif "VAL   -> Avg Seg:" in line:
        parts = line.split('|')
        s = parts[0].split(':')[1].strip()
        e = parts[1].split(':')[1].strip()
        t = parts[2].split(':')[1].strip()
        vals.append((s, e, t))

print("| Epoch | Train Total | Train Seg | Train Edge | Val Total | Val Seg | Val Edge |")
print("|---|---|---|---|---|---|---|")
for i in range(len(trains)):
    print(f"| {i+1} | {trains[i][2]} | {trains[i][0]} | {trains[i][1]} | {vals[i][2]} | {vals[i][0]} | {vals[i][1]} |")
