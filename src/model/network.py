import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.backbone import DualResNet34Backbone

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_p, None

class MVSSNetLite(nn.Module):
    def __init__(self):
        super(MVSSNetLite, self).__init__()
        self.backbone = DualResNet34Backbone()
        
        # Progressive upsampling blocks for decoder
        # Stage 4 to 3 (512 -> 256)
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3 to 2 (256 -> 128)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2 to 1 (128 -> 64)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Segmentation head outputs a 1-channel mask
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Edge head outputs a 1-channel edge map
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # Domain classifier on f4 (512 channels)
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, rgb_image, lambda_p=None):
        # Extract multi-scale fused features from backbone
        f1, f2, f3, f4 = self.backbone(rgb_image)
        
        # Domain classification branch (only if lambda_p is provided)
        domain_logits = None
        if lambda_p is not None:
            f4_grl = GradientReversalLayer.apply(f4, lambda_p)
            domain_logits = self.domain_classifier(f4_grl)

        # Decoder with progressive upsampling and skip connection concatenation
        x = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f3], dim=1)
        x = self.up_conv3(x)
        
        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f2], dim=1)
        x = self.up_conv2(x)
        
        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f1], dim=1)
        x = self.up_conv1(x)
        
        # Final upsample back to original image resolution
        x = F.interpolate(x, size=rgb_image.shape[2:], mode='bilinear', align_corners=False)
        
        seg_mask = self.seg_head(x)
        edge_map = self.edge_head(x)
        
        if lambda_p is not None:
            return seg_mask, edge_map, domain_logits
        return seg_mask, edge_map

if __name__ == '__main__':
    # Instantiate the model
    model = MVSSNetLite()
    
    # Dummy input tensors (batch=2, 3-channel, 224x224)
    rgb_input = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    seg_mask, edge_map, domain_logits = model(rgb_input, lambda_p=0.5)
    
    # Print shapes to confirm they match expected
    print("RGB Input shape:", rgb_input.shape)
    print("Segmentation Mask shape:", seg_mask.shape)
    print("Edge Map shape:", edge_map.shape)
    print("Domain Logits shape:", domain_logits.shape)
    
    assert seg_mask.shape == (2, 1, 224, 224), f"Expected seg_mask shape (2, 1, 224, 224), got {seg_mask.shape}"
    assert edge_map.shape == (2, 1, 224, 224), f"Expected edge_map shape (2, 1, 224, 224), got {edge_map.shape}"
    assert domain_logits.shape == (2, 2), f"Expected domain_logits shape (2, 2), got {domain_logits.shape}"
    print("Output shapes are correct.")
