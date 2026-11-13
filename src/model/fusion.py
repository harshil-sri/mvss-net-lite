import torch
import torch.nn as nn

"""
    CBAM fusion for MVSS-Net Lite. Contains architectural decisions made by Harshil:
    - Separate channel+spatial attention per branch (noise vs edge encode
      different information, so each learns its own weighting)
    - Fusion via concat + 1x1 conv rather than elementwise add (lets the
      network learn the blend rather than assuming equal contribution)
"""
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # squishes H,W down to 1,1
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # squeeze
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)    # excite
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.shape

        pooled = self.avg_pool(x)          # [B, C, 1, 1]
        pooled = pooled.view(b, c)         # [B, C]

        attn = self.mlp(pooled)            # [B, C]
        attn = self.sigmoid(attn)          # [B, C], values 0-1
        attn = attn.view(b, c, 1, 1)       # [B, C, 1, 1]

        out = x * attn                     # broadcast across H, W
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, H, W] -- this is the OUTPUT of ChannelAttention
        max_map = torch.max(x, dim=1, keepdim=True)[0]   # [B, 1, H, W]
        avg_map = torch.mean(x, dim=1, keepdim=True)      # [B, 1, H, W]
        combined = torch.cat([max_map, avg_map], dim=1)   # [B, 2, H, W]
        attn = self.conv(combined)                         # [B, 1, H, W]
        attn = self.sigmoid(attn)
        out = x * attn                                      # broadcast across C
        return out


class CBAMFusion(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn_noise = ChannelAttention(channels, reduction)
        self.spatial_attn_noise = SpatialAttention(kernel_size)
        self.channel_attn_edge = ChannelAttention(channels, reduction)
        self.spatial_attn_edge = SpatialAttention(kernel_size)
        self.fuse_conv = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, noise_feat, edge_feat):
        # noise_feat, edge_feat: both [B, C, H, W]
        n = self.channel_attn_noise(noise_feat)
        n = self.spatial_attn_noise(n)

        e = self.channel_attn_edge(edge_feat)
        e = self.spatial_attn_edge(e)

        combined = torch.cat([n, e], dim=1)   # [B, 2C, H, W]
        fused = self.fuse_conv(combined)      # [B, C, H, W]

        return fused


if __name__ == "__main__":
    # sanity check
    noise = torch.randn(2, 128, 28, 28)
    edge = torch.randn(2, 128, 28, 28)
    fusion = CBAMFusion(channels=128)
    out = fusion(noise, edge)
    print(out.shape)  # expect: torch.Size([2, 128, 28, 28])