import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

__all__ = [
    'SelfAttention', 'ChannelAttention',
    'SpatialAttention', 'CBAM',
    'CrossModalAttention'
]

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        """
        Initialize the SelfAttention module.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = (in_channels // 8) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        q = einops.rearrange(self.query(x), 'b c h w -> b (h w) c')
        k = einops.rearrange(self.key(x), 'b c h w -> b c (h w)')
        v = einops.rearrange(self.value(x), 'b c h w -> b c (h w)')

        attn = F.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = einops.rearrange(torch.bmm(v, attn.permute(0, 2, 1)), 'b c (h w) -> b c h w', h=h, w=w)

        return self.gamma * out + x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int)-> None:
        """
        Initialize the ChannelAttention module.

        Args:
            in_channels (int): Number of input channels.
            reduction_ratio (int): Ratio for reducing channels in the MLP.
        """
        super().__init__()
        self.sigmoid= nn.Sigmoid()
        self.mlp= nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels//reduction_ratio),
            nn.ReLU(inplace= True),
            nn.Linear(in_channels//reduction_ratio, in_channels)
        )
        self.mean_pool= nn.AdaptiveAvgPool2d((1))
        self.max_pool = nn.AdaptiveMaxPool2d((1))

    def forward(self, x: torch.Tensor)->torch.Tensor:
        max_p= self.max_pool(x)
        mean_p= self.mean_pool(x)
        attention_map= self.sigmoid(self.mlp(max_p)+ self.mlp(mean_p))

        return x* attention_map.unsqueeze(2).unsqueeze(3)

class SpatialAttention(nn.Module):
    def __init__(self)-> None:
        super().__init__()
        self.conv= nn.Conv2d(2, 1, kernel_size= 7, padding=3)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        max_p= torch.max(x, dim=1, keepdim=True)[0]
        mean_p= torch.mean(x, dim=1, keepdim=True)

        pooled_features= torch.cat([mean_p, max_p], dim=1)

        attention_map= self.conv(pooled_features)
        attention_map= self.sigmoid(attention_map)

        return x* attention_map
    
class CBAM(nn.Module):
    def __init__(self, in_channels:int, r:int= 8) -> None:
        """
        Initialize the CBAM module (Channel and Spatial Attention Module).

        Args:
            in_channels (int): Number of input channels.
            reduction_ratio (int): Ratio for reducing channels in the ChannelAttention module. Default is 8.
        """
        super().__init__()
        self.channel_attention= ChannelAttention(in_channels, r)
        self.spatial_attention= SpatialAttention()

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x= self.channel_attention(x)
        x= self.spatial_attention(x)
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, dim: int):
        """
        Initialize the CrossModalAttention module.

        Args:
            dim (int): Dimensionality of input features for query, key, and value.
        """
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        return out