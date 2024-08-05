import torch
import torch.nn as nn
import torch.nn.functional as F

__all__= [
    'DoubleConvBlock',
    'TransposedConvBlock',
    'ResidualDecoderBlock',
    'FPNBlock'
]

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the DoubleConvBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        self.double_conv= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return self.double_conv(x)

class TransposedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the TransposedConvBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 2, stride= 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.deconv(x)
        if skip is not None:
            x = x + skip
        return F.relu(x)
    
class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels: int , out_channels: int, use_skip: bool= False)-> None:
        """
        Initialize the ResidualDecoderBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_skip (bool): Whether to use skip connections. Default is False.
        """
        super().__init__()

        self.use_skip= use_skip
        if use_skip:
            self.upsample= nn.Upsample(scale_factor= 2, mode= 'bilinear', align_corners= True)

        self.channel_reduction= nn.Conv2d(in_channels, out_channels, kernel_size= 1)
        self.double_conv= DoubleConvBlock(in_channels, out_channels)
        self.relu = nn.ReLU(inplace= True)
        self.convT = nn.ConvTranspose2d(out_channels, out_channels, kernel_size= 2, stride= 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor= None)-> torch.Tensor:
        if self.use_skip:
            x= x+ self.upsample(skip)
        
        res= self.channel_reduction(x)
        x= self.double_conv(x)
        x= self.relu(x+ res)
        return self.convT(x)
    
class FPNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the FPNBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        self.adjust= nn.Upsample(scale_factor= 2, mode= 'nearest')
        self.conv= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor, skip: torch.Tensor= None)-> torch.Tensor:
        if skip is not None:
            if x.shape != skip.shape:
                x= self.adjust(x)
            x= x+ skip
        return F.relu(self.conv(x))