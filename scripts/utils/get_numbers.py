import json, glob

def check_json(file, keys):
    try:
        with open(file, 'r') as f:
            data = str(f.read())
            found = []
            for k in keys:
                if str(k) in data:
                    found.append(k)
            if found:
                print(f"{file}: {found}")
    except:
        pass

keys = [66.9, 39.4, 96.4, 100.0, 0.0, 0.825, 0.829, 0.834, 0.804, 0.784, 0.807, 0.838, 0.864, 0.361, 0.267, 0.405, 0.450, 0.521, 0.554, 0.780, 0.636, 0.872, 0.17]
for f in glob.glob("reports/*.json"):
    check_json(f, keys)

