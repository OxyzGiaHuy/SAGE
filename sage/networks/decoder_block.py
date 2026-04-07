import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    Single decoder block for UNet-style upsampling.

    This block performs:
    1. Upsampling of the input feature map (2x)
    2. Concatenation with skip connection from encoder
    3. Two 3x3 convolutions with batch norm and ReLU
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )

        combined_channels = in_channels + skip_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder block.

        Args:
            x (torch.Tensor): Input from previous layer (B, in_channels, H, W)
            skip (torch.Tensor): Skip connection from encoder (B, skip_channels, 2H, 2W)

        Returns:
            torch.Tensor: Output features (B, out_channels, 2H, 2W)
        """
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
