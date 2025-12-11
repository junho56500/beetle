import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any

# Placeholder for a generic ConvModule.
# In a real scenario, this would be imported from a library like MMDetection.
class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False,
                 norm_cfg: Optional[Dict] = None, act_cfg: Optional[Dict] = None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels) if norm_cfg else None
        self.act = nn.ReLU() if act_cfg else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

# Placeholder for a generic build_attention function.
# This would typically create a self-attention, non-local, or other attention mechanism.
def build_attention(attention_cfg: Dict[str, Any]) -> nn.Module:
    """
    Builds an attention layer based on the provided configuration.
    This is a simplified example; a real implementation would be more complex.
    """
    if attention_cfg['type'] == 'SelfAttention':
        # Example of a simple self-attention block
        in_channels = attention_cfg['in_channels']
        inter_channels = in_channels // 8
        return nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.Sigmoid() # Use Sigmoid for channel-wise attention or spatial weights
        )
    elif attention_cfg['type'] == 'ChannelAttention':
        # Simple global average pooling + FC layers for channel attention
        in_channels = attention_cfg['in_channels']
        reduction_ratio = attention_cfg.get('reduction_ratio', 16)
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Unsupported attention type: {attention_cfg['type']}")


class DownsampleModule(nn.Module):
    """
    A module for downsampling feature maps, incorporating a ConvModule,
    an additional convolutional layer, and an attention mechanism.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after downsampling.
        conv_cfg (Optional[Dict]): Configuration for the initial ConvModule.
        attention_cfg (Optional[Dict]): Configuration for the attention layer.
        stride (int): Stride for the downsampling convolution. Defaults to 2.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: Optional[Dict] = None,
                 attention_cfg: Optional[Dict] = None,
                 stride: int = 2):
        super().__init__()

        # 1. Initial ConvModule for feature processing
        # This can be used for initial feature extraction before downsampling.
        # Assuming a basic configuration if not provided.
        _conv_cfg = conv_cfg if conv_cfg else dict(
            kernel_size=3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.conv_module = ConvModule(
            in_channels, out_channels, stride=1, **_conv_cfg)

        # 2. Additional Convolutional Layer for downsampling
        # This layer performs the actual downsampling, often with stride=2.
        self.downsample_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.downsample_norm = nn.BatchNorm2d(out_channels) # Add batch norm after downsample conv
        self.downsample_act = nn.ReLU(inplace=True) # Add activation after batch norm


        # 3. Attention Layer
        self.attention = None
        if attention_cfg:
            # Update attention_cfg with in_channels if not already present,
            # assuming attention operates on the output of downsample_conv
            if 'in_channels' not in attention_cfg:
                attention_cfg['in_channels'] = out_channels
            self.attention = build_attention(attention_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DownsampleModule.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Downsampled feature map with attention applied.
        """
        # Pass through the initial ConvModule
        x = self.conv_module(x)

        # Apply the additional convolutional layer for downsampling
        x = self.downsample_conv(x)
        x = self.downsample_norm(x)
        x = self.downsample_act(x)

        # Apply attention if configured
        if self.attention:
            # For channel attention, we multiply the original feature map
            # with the attention weights (e.g., [N, C, 1, 1])
            if self.attention.__class__.__name__ == 'Sequential' and \
               isinstance(self.attention[-1], nn.Sigmoid): # Heuristic for attention weights
                attention_weights = self.attention(x)
                x = x * attention_weights # Apply channel attention
            else:
                # For other types of attention (e.g., spatial),
                # the attention layer might return a modified feature map directly
                x = self.attention(x) + x # Residual connection for attention is common

        return x

if __name__ == '__main__':
    # Example Usage:
    # Define input tensor
    input_tensor = torch.randn(1, 64, 32, 32) # Batch, Channels, Height, Width

    print(f"Input tensor shape: {input_tensor.shape}")

    # Example 1: Downsample with Channel Attention
    downsample_module_channel_att = DownsampleModule(
        in_channels=64,
        out_channels=128,
        conv_cfg=dict(kernel_size=3, padding=1, norm_cfg=dict(type='BN2d'), act_cfg=dict(type='ReLU')),
        attention_cfg=dict(type='ChannelAttention', reduction_ratio=8),
        stride=2
    )
    output_channel_att = downsample_module_channel_att(input_tensor)
    print(f"Output with Channel Attention shape: {output_channel_att.shape}") # Expected: (1, 128, 16, 16)

    # Example 2: Downsample with a simplified Self-Attention (as placeholder)
    downsample_module_self_att = DownsampleModule(
        in_channels=64,
        out_channels=128,
        conv_cfg=dict(kernel_size=3, padding=1, norm_cfg=dict(type='BN2d'), act_cfg=dict(type='ReLU')),
        attention_cfg=dict(type='SelfAttention', in_channels=128), # in_channels must match out_channels of downsample_conv
        stride=2
    )
    output_self_att = downsample_module_self_att(input_tensor)
    print(f"Output with Self-Attention shape: {output_self_att.shape}") # Expected: (1, 128, 16, 16)

    # Example 3: Downsample without Attention
    downsample_module_no_att = DownsampleModule(
        in_channels=64,
        out_channels=128,
        conv_cfg=dict(kernel_size=3, padding=1, norm_cfg=dict(type='BN2d'), act_cfg=dict(type='ReLU')),
        attention_cfg=None, # No attention layer
        stride=2
    )
    output_no_att = downsample_module_no_att(input_tensor)
    print(f"Output without Attention shape: {output_no_att.shape}") # Expected: (1, 128, 16, 16)
