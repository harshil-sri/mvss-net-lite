import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ResNetUNetFPN(nn.Module):
    def __init__(self):
        super(ResNetUNetFPN, self).__init__()
        resnet = resnet34(weights='IMAGENET1K_V1')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.up_conv3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, rgb_image):
        x = self.conv1(rgb_image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        x = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f3], dim=1)
        x = self.up_conv3(x)

        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f2], dim=1)
        x = self.up_conv2(x)

        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f1], dim=1)
        x = self.up_conv1(x)

        x = F.interpolate(x, size=rgb_image.shape[2:], mode='bilinear', align_corners=False)

        seg_mask = self.seg_head(x)
        edge_map = self.edge_head(x)

        return seg_mask, edge_map


if __name__ == '__main__':
    model = ResNetUNetFPN()
    rgb_input = torch.randn(2, 3, 224, 224)
    seg_mask, edge_map = model(rgb_input)
    print("RGB Input shape:", rgb_input.shape)
    print("Segmentation Mask shape:", seg_mask.shape)
    print("Edge Map shape:", edge_map.shape)
    assert seg_mask.shape == (2, 1, 224, 224), f"Expected (2,1,224,224), got {seg_mask.shape}"
    assert edge_map.shape == (2, 1, 224, 224), f"Expected (2,1,224,224), got {edge_map.shape}"
    print("Output shapes are correct.")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")