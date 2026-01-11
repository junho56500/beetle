import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class FPN(nn.Module):
    """
    A simplified FPN module that takes feature maps from a backbone
    and generates a feature pyramid.
    """
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # Top-down path convolutions
        # We use 1x1 convolutions to match the number of channels of the bottom-up features
        self.conv_lateral_p5 = nn.Conv2d(in_channels_list[3], out_channels, kernel_size=1)
        self.conv_lateral_p4 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.conv_lateral_p3 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.conv_lateral_p2 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)

        # Convolutions to smooth the fused feature maps
        self.conv_smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_smooth_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, backbone_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            backbone_features (List[torch.Tensor]): A list of feature maps from
                                                    the bottom-up path of a backbone,
                                                    ordered from shallowest to deepest.
                                                    e.g., [C2, C3, C4, C5]
        Returns:
            List[torch.Tensor]: A list of FPN feature maps (P2, P3, P4, P5),
                                ordered from finest to coarsest resolution.
        """
        # The input backbone features are typically:
        # C2: [N, in_channels_list[0], H/4, W/4]
        # C3: [N, in_channels_list[1], H/8, W/8]
        # C4: [N, in_channels_list[2], H/16, W/16]
        # C5: [N, in_channels_list[3], H/32, W/32]
        
        c2, c3, c4, c5 = backbone_features

        # --- Top-down path + Lateral connections ---
        
        # P5 is simply the C5 feature map after a 1x1 conv
        p5 = self.conv_lateral_p5(c5)

        # Upsample P5 and add it to C4
        upsampled_p5 = F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.conv_lateral_p4(c4) + upsampled_p5

        # Upsample P4 and add it to C3
        upsampled_p4 = F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.conv_lateral_p3(c3) + upsampled_p4

        # Upsample P3 and add it to C2
        upsampled_p3 = F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.conv_lateral_p2(c2) + upsampled_p3

        # --- Final smoothing of the feature maps ---
        p5 = self.conv_smooth_p5(p5)
        p4 = self.conv_smooth_p4(p4)
        p3 = self.conv_smooth_p3(p3)
        p2 = self.conv_smooth_p2(p2)
        
        # Return the feature pyramid
        return [p2, p3, p4, p5]

# --- Example of a Mock Backbone ---
# This class simulates a ResNet backbone's output
class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1) # Initial layer

    def forward(self, x):
        # Simulate feature maps at different strides (4, 8, 16, 32)
        # In a real ResNet, these would be the outputs of different stages
        h, w = x.shape[2], x.shape[3]
        c2 = torch.randn(x.shape[0], 64, h // 4, w // 4)
        c3 = torch.randn(x.shape[0], 128, h // 8, w // 8)
        c4 = torch.randn(x.shape[0], 256, h // 16, w // 16)
        c5 = torch.randn(x.shape[0], 512, h // 32, w // 32)
        return [c2, c3, c4, c5]


if __name__ == '__main__':
    # Define FPN parameters
    # The input channels should match the output channels of your backbone stages
    backbone_in_channels = [64, 128, 256, 512]
    # The output channel count for each FPN level is typically fixed
    fpn_out_channels = 256

    # Create model instances
    mock_backbone = MockBackbone()
    fpn_model = FPN(in_channels_list=backbone_in_channels, out_channels=fpn_out_channels)

    # Create a dummy input image
    dummy_image = torch.randn(2, 3, 256, 256) # Batch=2, Channels=3, Size=256x256

    # 1. Get features from the backbone
    backbone_features = mock_backbone(dummy_image)
    print("--- Backbone Feature Map Shapes ---")
    for i, features in enumerate(backbone_features):
        print(f"C{i+2} Shape: {features.shape}")

    # 2. Pass them to the FPN
    fpn_features = fpn_model(backbone_features)

    print("\n--- FPN Feature Map Shapes ---")
    for i, features in enumerate(fpn_features):
        print(f"P{i+2} Shape: {features.shape}")

    print("\nFPN successfully created a multi-scale feature pyramid!")
    print("Each output level has the same number of channels (256) but a different spatial resolution.")