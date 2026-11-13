import torch
import torch.nn as nn

class ConstrainedConv2d(nn.Conv2d):
    """
    Constrained convolution layer for noise feature extraction.
    
    IMPORTANT: You must call `normalize_weights()` on this layer (or its parent module)
    after every optimizer.step() during training to enforce the constraint!
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.normalize_weights()
        
    def normalize_weights(self):
        with torch.no_grad():
            c_h, c_w = self.kernel_size[0] // 2, self.kernel_size[1] // 2
            
            # Temporarily zero out the center weights
            self.weight.data[:, :, c_h, c_w] = 0.0
            
            # Sum over spatial dimensions (so each 2D kernel slice normalizes independently)
            sums = self.weight.data.sum(dim=(2, 3), keepdim=True)
            
            # Avoid division by zero
            sums[sums == 0] = 1.0
            
            # Normalize so surrounding weights sum to +1
            self.weight.data /= sums
            
            # Fix center weights to -1
            self.weight.data[:, :, c_h, c_w] = -1.0


class NoiseBranchInput(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5):
        super().__init__()
        self.constrained_conv = ConstrainedConv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            bias=False
        )
        
    def forward(self, x):
        return self.constrained_conv(x)
        
    def normalize_weights(self):
        self.constrained_conv.normalize_weights()


if __name__ == "__main__":
    model = NoiseBranchInput(in_channels=3, out_channels=3)
    
    dummy_img = torch.randn(2, 3, 224, 224)
    out = model(dummy_img)
    print("Output shape:", out.shape)
    
    # Simulate an optimizer update that messes up the weights
    model.constrained_conv.weight.data += 0.5
    
    # Enforce constraints
    model.normalize_weights()
    
    c_h, c_w = model.constrained_conv.kernel_size[0] // 2, model.constrained_conv.kernel_size[1] // 2
    
    # Check center weight of filter 0, in_channel 0
    center_wt = model.constrained_conv.weight.data[0, 0, c_h, c_w].item()
    
    # Check sum of surrounding weights for filter 0, in_channel 0
    total_sum = model.constrained_conv.weight.data[0, 0].sum().item()
    surrounding_sum = total_sum - center_wt
    
    print(f"Center weight: {center_wt}")
    print(f"Sum of surrounding weights (filter 0, in_channel 0): {surrounding_sum:.4f}")
