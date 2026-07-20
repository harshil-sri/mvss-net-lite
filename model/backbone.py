import torch
import torch.nn as nn
from torchvision.models import resnet34

from model.fusion import CBAMFusion
from model.constrained_conv import NoiseBranchInput

class DualResNet34Backbone(nn.Module):
    def __init__(self):
        super(DualResNet34Backbone, self).__init__()
        
        # Load two standard resnet34 stems (BasicBlock architecture)
        noise_resnet = resnet34(pretrained=False)
        edge_resnet = resnet34(pretrained=False)
        
        # Noise branch components
        self.noise_extractor = NoiseBranchInput()
        self.noise_conv1 = noise_resnet.conv1
        self.noise_bn1 = noise_resnet.bn1
        self.noise_relu = noise_resnet.relu
        self.noise_maxpool = noise_resnet.maxpool
        self.noise_layer1 = noise_resnet.layer1
        self.noise_layer2 = noise_resnet.layer2
        self.noise_layer3 = noise_resnet.layer3
        self.noise_layer4 = noise_resnet.layer4

        # Edge branch components
        self.edge_conv1 = edge_resnet.conv1
        self.edge_bn1 = edge_resnet.bn1
        self.edge_relu = edge_resnet.relu
        self.edge_maxpool = edge_resnet.maxpool
        self.edge_layer1 = edge_resnet.layer1
        self.edge_layer2 = edge_resnet.layer2
        self.edge_layer3 = edge_resnet.layer3
        self.edge_layer4 = edge_resnet.layer4
        
        # Fusion modules for each of the 4 stages
        # ResNet-34 stage channels: 64, 128, 256, 512
        self.fuse1 = CBAMFusion(channels=64)
        self.fuse2 = CBAMFusion(channels=128)
        self.fuse3 = CBAMFusion(channels=256)
        self.fuse4 = CBAMFusion(channels=512)

    def forward(self, rgb_input):
        # Noise branch feature extraction
        noise_input = self.noise_extractor(rgb_input)
        n = self.noise_conv1(noise_input)
        n = self.noise_bn1(n)
        n = self.noise_relu(n)
        n = self.noise_maxpool(n)
        
        n1 = self.noise_layer1(n)
        n2 = self.noise_layer2(n1)
        n3 = self.noise_layer3(n2)
        n4 = self.noise_layer4(n3)

        # Edge branch feature extraction
        e = self.edge_conv1(rgb_input)
        e = self.edge_bn1(e)
        e = self.edge_relu(e)
        e = self.edge_maxpool(e)
        
        e1 = self.edge_layer1(e)
        e2 = self.edge_layer2(e1)
        e3 = self.edge_layer3(e2)
        e4 = self.edge_layer4(e3)

        # Cross-branch fusion at each stage
        f1 = self.fuse1(n1, e1)
        f2 = self.fuse2(n2, e2)
        f3 = self.fuse3(n3, e3)
        f4 = self.fuse4(n4, e4)
        
        return f1, f2, f3, f4
