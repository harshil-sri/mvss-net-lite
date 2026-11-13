import json, glob, re

keys = ["66.9", "39.4", "96.4", "100.0", "0.0", "0.825", "0.829", "0.834", "0.804", "0.784", "0.807", "0.838", "0.864", "0.361", "0.267", "0.405", "0.450", "0.521", "0.554", "0.780", "0.636", "0.872", "0.17"]
keys_float = [float(k) for k in keys]

for f in glob.glob("reports/*.json"):
    if "manifest" in f: continue
    try:
        with open(f, 'r') as fp:
            data = fp.read()
            # find all numbers
            nums = re.findall(r'\b0\.\d+|\b\d+\.\d+|\b100\.0\b|\b0\.0\b', data)
            nums_float = [float(n) for n in nums]
            found = []
            for k in keys_float:
                if any(abs(n - k) < 1e-4 for n in nums_float):
                    found.append(k)
            if found:
                print(f"{f}: {list(set(found))}")
    except Exception as e:
        pass

