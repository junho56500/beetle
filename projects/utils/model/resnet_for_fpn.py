import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# (Paste the Bottleneck class and ResNet class here from above)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block: nn.Module, layers: List[int], num_classes: int = 1000):
        super().__init__()
        self.inplanes = 64
        
        #input : (N, 3, 224, 224)
        #STEM output : (N, 64, 56, 56)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #layer1 output : (N, 256, 56, 56) by expansion : 4
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        #layer2 output : (N, 512, 28, 28)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        #layer3 output : (N, 1024, 14, 14)       
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        #layer4 output : (N, 2048, 7, 7)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


# The FPN class from previous discussion
class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.conv_lateral_p5 = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=1)
        self.conv_lateral_p4 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.conv_lateral_p3 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.conv_lateral_p2 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.conv_smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_smooth_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, backbone_features: List[torch.Tensor]) -> List[torch.Tensor]:
        c2, c3, c4, c5 = backbone_features
        #input : (N, 2048, 7, 7)
        
        #Output Shape (P5): (N, 256, 7, 7)
        p5 = self.conv_lateral_p5(c5)
        
        # P4 (N, 256, 14, 14) : C4 ((N, 1024, 14, 14)) with the upsampled P5 ((N, 256, 14, 14)).
        upsampled_p5 = F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.conv_lateral_p4(c4) + upsampled_p5
        
        # P4 (N, 256, 28, 28) : C3 ((N, 512, 28, 28)) with the upsampled P4 ((N, 256, 28, 28)).
        upsampled_p4 = F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.conv_lateral_p3(c3) + upsampled_p4
        
        # P2 (N, 256, 56, 56) : C2 ((N, 256, 56, 56)) with the upsampled P3 ((N, 256, 56, 56)).
        upsampled_p3 = F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.conv_lateral_p2(c2) + upsampled_p3
        p5 = self.conv_smooth_p5(p5)
        p4 = self.conv_smooth_p4(p4)
        p3 = self.conv_smooth_p3(p3)
        p2 = self.conv_smooth_p2(p2)
        return [p2, p3, p4, p5]


# Main execution block
if __name__ == '__main__':
    # ResNet-50 uses [3, 4, 6, 3] Bottleneck blocks
    num_blocks_per_stage_50 = [3, 4, 6, 3]
    num_classes_dummy = 1000

    # 1. Create the ResNet-50 backbone
    # The forward pass of this ResNet is modified to return features from all layers
    resnet50_backbone = ResNet(Bottleneck, num_blocks_per_stage_50, num_classes_dummy)

    # 2. Get the output channel list for the FPN
    # ResNet-50's layers output: layer1 (C2) -> 256, layer2 (C3) -> 512, layer3 (C4) -> 1024, layer4 (C5) -> 2048
    resnet_out_channels = [64 * Bottleneck.expansion, 128 * Bottleneck.expansion, 256 * Bottleneck.expansion, 512 * Bottleneck.expansion]
    # For ResNet-50: [256, 512, 1024, 2048]
    fpn_out_channels = 256 # A common fixed channel count for FPN levels

    # 3. Create the FPN module
    fpn_model = FPN(in_channels_list=resnet_out_channels, out_channels=fpn_out_channels)

    # 4. Create a dummy input image  image size : 224,224, 2 images (batch size), 3 channels 
    dummy_input_image = torch.randn(2, 3, 224, 224)

    # --- Run the combined forward pass ---
    # a) Get backbone features from ResNet
    backbone_features = resnet50_backbone(dummy_input_image)
    
    print("--- ResNet-50 Backbone Feature Map Shapes ---")
    for i, features in enumerate(backbone_features):
        print(f"C{i+2} Shape: {features.shape}")

    # b) Pass these features to the FPN
    fpn_features = fpn_model(backbone_features)

    print("\n--- FPN Feature Map Shapes ---")
    for i, features in enumerate(fpn_features):
        print(f"P{i+2} Shape: {features.shape}")

    print("\nResNet-50 backbone successfully generated features for the FPN!")