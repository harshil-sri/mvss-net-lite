import os
import json
import time
import torch
import numpy as np

from model.network import MVSSNetLite
from model.pathb_network import ResNetUNetFPN

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def calculate_flops(model, input_shape=(1, 3, 256, 256)):
    model_cpu = model.cpu()
    model_cpu.eval()
    dummy_input = torch.randn(input_shape)
    
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model_cpu, dummy_input)
        return flops.total() / 1e9  # GFLOPs
    except Exception as e:
        print(f"fvcore failed: {e}")
        try:
            from thop import profile
            macs, params = profile(model_cpu, inputs=(dummy_input, ), verbose=False)
            return (macs * 2) / 1e9 # approx GFLOPS
        except Exception as e2:
            print(f"thop failed: {e2}")
            return -1

def benchmark_inference(model, device, batch_size, input_shape=(3, 256, 256), warmup=10, iters=100):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(batch_size, *input_shape).to(device)
    
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
        
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)
            
    times = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.time()
            model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000) # in ms
            
    mean_time = np.mean(times)
    std_time = np.std(times)
    img_per_sec = (batch_size / (mean_time / 1000.0))
    
    return mean_time, std_time, img_per_sec

def get_peak_memory(model, device, input_shape=(1, 3, 256, 256)):
    if not torch.cuda.is_available():
        return 0.0
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        model(dummy_input)
    
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 * 1024)

def benchmark_model(model, name, device):
    print(f"Benchmarking {name}...")
    model.eval()
    
    # Params
    total_params, trainable_params = count_params(model)
    
    # FLOPs
    gflops = calculate_flops(model)
    
    # Inference speed
    bs1_mean, _, _ = benchmark_inference(model, device, 1)
    bs4_mean, _, _ = benchmark_inference(model, device, 4)
    bs8_mean, _, img_per_sec_bs8 = benchmark_inference(model, device, 8)
    
    # Memory
    gpu_mem_mb = get_peak_memory(model, device)
    
    return {
        "params": total_params,
        "trainable_params": trainable_params,
        "gflops": gflops,
        "inference": {
            "bs1_ms": bs1_mean,
            "bs4_ms": bs4_mean,
            "bs8_ms": bs8_mean,
            "img_per_sec_bs8": img_per_sec_bs8
        },
        "gpu_mem_mb": gpu_mem_mb
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    # Initialize models
    mvss = MVSSNetLite()
    resnet_unet = ResNetUNetFPN()
    
    results["mvss_net_lite"] = benchmark_model(mvss, "mvss_net_lite", device)
    results["resnet_unet_fpn"] = benchmark_model(resnet_unet, "resnet_unet_fpn", device)
    
    # Comparisons
    param_ratio = results["mvss_net_lite"]["params"] / results["resnet_unet_fpn"]["params"] if results["resnet_unet_fpn"]["params"] > 0 else 0
    speed_bs8_ratio = results["mvss_net_lite"]["inference"]["bs8_ms"] / results["resnet_unet_fpn"]["inference"]["bs8_ms"] if results["resnet_unet_fpn"]["inference"]["bs8_ms"] > 0 else 0
    
    results["comparison"] = {
        "param_ratio": param_ratio,
        "speed_bs8_ratio": speed_bs8_ratio
    }
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/model_benchmarks.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
