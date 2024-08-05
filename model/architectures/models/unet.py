import torch
import torch.nn as nn
from torchvision import models
from blocks import ResidualDecoderBlock, FPNBlock, TransposedConvBlock, SelfAttention, CBAM
from transformers import ViTModel, ViTConfig

__all__= ['EfficientUNet', 'EfficientUNet_v2', 'ViTUNet' ]

class EfficientUNet(nn.Module):
    def __init__(self)-> None:
        super().__init__()
        
        self.encoder = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        ).features
        
        self.decoder1 = ResidualDecoderBlock(1280, 640)
        self.decoder2 = ResidualDecoderBlock(640, 320, use_skip= True)
        self.decoder3 = ResidualDecoderBlock(320, 160, use_skip= True)
        self.decoder4 = ResidualDecoderBlock(160, 64, use_skip= True)

        self.attention1= CBAM(640)
        self.attention2= CBAM(320)
        self.attention3= CBAM(160)
        self.attention4= CBAM(64)
        
        self.adjust_enc4 = nn.Conv2d(1280, 640, kernel_size=1)
        self.adjust_enc3 = nn.Conv2d(160, 320, kernel_size=1)
        self.adjust_enc2 = nn.Conv2d(64, 160, kernel_size=1)
        self.adjust_out= nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x= self.encoder[0:2](x)
        enc2= self.encoder[2:4](x)
        enc3= self.encoder[4:6](enc2)
        enc4= self.encoder[6:8](enc3)
        enc5= self.encoder[8:](enc4)

        x= self.decoder1(enc5)
        x= self.attention1(x)

        x= self.decoder2(x, self.adjust_enc4(enc4))
        x= self.attention2(x)

        x= self.decoder3(x, self.adjust_enc3(enc3))
        x=self.attention3(x)

        x= self.decoder4(x, self.adjust_enc2(enc2))
        x=self.attention4(x)

        return self.adjust_out(x)
    
class EfficientUNet_v2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder= models.efficientnet_b0(
            weights= models.EfficientNet_B0_Weights.IMAGENET1K_V1
        ).features
        

        self.self_attention= SelfAttention(1280)
        self.bottleneck= nn.Sequential(
            nn.Conv2d(1280, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace= True),
            nn.Conv2d(320, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace= True)
        )
        self.dropout = nn.Dropout(0.5)

        self.decoder= nn.ModuleList([
            FPNBlock(320, 192),
            FPNBlock(192, 112),
            FPNBlock(112, 80),
            FPNBlock(80, 40),
            FPNBlock(40, 32)
        ])

        self.adjust_out= nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        skips=[]
        for i, layer in enumerate(self.encoder):
            x= layer(x)
            if i in [3, 4, 5, 6, 7]:
                skips.append(x)
        
        x= self.self_attention(x)
        x= self.bottleneck(x)
        x= self.dropout(x)

        for i, layer in enumerate(self.decoder):
            x= layer(x, skips[-(i+1)])

        return self.adjust_out(x)

class ViTUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit = ViTModel(ViTConfig(image_size=448))

        self.self_attention = SelfAttention(self.vit.config.hidden_size)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.vit.config.hidden_size, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
            nn.Conv2d(320, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(0.5)

        self.decoder = nn.ModuleList([
            TransposedConvBlock(320, 192),
            TransposedConvBlock(192, 112),
            TransposedConvBlock(112, 80),
            TransposedConvBlock(80, 40),
            TransposedConvBlock(40, 32)
        ])

        self.adjust_out = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.ffn= nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace= True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace= True),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x).last_hidden_state[:, 1:, :]

        b, hw, c= x.shape
        h= w = int((hw)**0.5)
        x = x.permute(0, 2, 1).view(b, c, h, w)

        x = self.self_attention(x)
        x = self.bottleneck(x)
        x = self.dropout(x)

        for layer in self.decoder:
            x = layer(x)
        return self.adjust_out(x)