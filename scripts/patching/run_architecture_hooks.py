import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/home/harshil/mvss-net-lite')

try:
    from model.network import MVSSNetLite
    model = MVSSNetLite()
    
    with open('handoff/model_print.txt', 'w') as f:
        f.write(str(model))
        
    params = sum(p.numel() for p in model.parameters())
    
    print(f"Params: {params}")

    hooks_data = []

    def hook_fn(name):
        def fn(module, inp, outp):
            if isinstance(outp, torch.Tensor):
                shape = list(outp.shape)
            elif isinstance(outp, (list, tuple)):
                shape = [list(o.shape) for o in outp]
            else:
                shape = "Unknown"
            hooks_data.append(f"{name}: {shape}")
        return fn

    # We want to hook:
    # Bayar output, stems, encoder stages, fusions, decoder steps, logits
    # We will just hook modules by name.
    
    for name, module in model.named_modules():
        module.register_forward_hook(hook_fn(name))

    x = torch.randn(1, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        out = model(x)
        
    with open('handoff/hooks_output.txt', 'w') as f:
        for line in hooks_data:
            f.write(line + '\n')
            
except Exception as e:
    import traceback
    traceback.print_exc()

