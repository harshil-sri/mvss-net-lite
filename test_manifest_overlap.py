import json

def test_overlap():
    with open('reports/manifest.json', 'r') as f:
        manifest = json.load(f)
        
    train_paths = set(s['image'] for s in manifest['train'])
    val_paths = set(s['image'] for s in manifest['val'])
    test_paths = set(s['image'] for s in manifest['test'])
    
    print(f"Train sizes: {len(train_paths)}")
    print(f"Val sizes:   {len(val_paths)}")
    print(f"Test sizes:  {len(test_paths)}")
    
    train_val = train_paths.intersection(val_paths)
    train_test = train_paths.intersection(test_paths)
    val_test = val_paths.intersection(test_paths)
    
    print("\n--- OVERLAP CHECK ---")
    print(f"Train & Val Overlap:  {len(train_val)}")
    print(f"Train & Test Overlap: {len(train_test)}")
    print(f"Val & Test Overlap:   {len(val_test)}")

if __name__ == '__main__':
    test_overlap()
