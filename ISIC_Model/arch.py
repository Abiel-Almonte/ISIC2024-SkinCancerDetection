import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import Tuple

class DoubleConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, r: int)-> None:
        super().__init__()
        self.sigmoid= nn.Sigmoid()
        self.mlp= nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels//r),
            nn.ReLU(inplace= True),
            nn.Linear(in_channels//r, in_channels)
        )
        self.mean_pool= nn.AdaptiveAvgPool2d((1))
        self.max_pool = nn.AdaptiveMaxPool2d((1))

    def forward(self, x: torch.Tensor)->torch.Tensor:
        max_p= self.max_pool(x)
        mean_p= self.mean_pool(x)
        attention_map= self.sigmoid(self.mlp(max_p) + self.mlp(mean_p))
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
    def __init__(self, in_channels:int, r:int) -> None:
        super().__init__()
        self.channel_attention= ChannelAttention(in_channels, r)
        self.spatial_attention= SpatialAttention()

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x= self.channel_attention(x)
        x= self.spatial_attention(x)
        return x
        
class DecoderBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int)-> None:
        super().__init__()

        self.adjust_residual= nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, 2, stride= 2)
        self.double_conv= DoubleConv(out_channels, out_channels)
        self.relu = nn.ReLU(inplace= True)
        self.attention= CBAM(out_channels, 16)

    def forward(self, x:torch.Tensor, skip:torch.Tensor= None)-> torch.Tensor:
        res= self.adjust_residual(x)

        x= self.convT(x)
        x= self.double_conv(x+ skip)
        x= self.attention(x + res)
        return self.relu(x )
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        self.double_conv= DoubleConv(in_channels, out_channels)
        self.pool=nn.MaxPool2d(kernel_size=2)

    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        x= self.double_conv(x)
        pool= self.pool(x)

        return x, pool
    
class EfficientUNet(nn.Module):
    def __init__(self)-> None:
        super().__init__()
        
        #EfficientNet encoder
        #self.encoder = models.efficientnet_v2_s(
            #weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        #).features

        #for param in self.encoder.parameters():
            #param.requires_grad= False

        #U-Net encoder
        self.encoder1= EncoderBlock(3, 64)
        self.encoder2= EncoderBlock(64, 128)
        self.encoder3= EncoderBlock(128, 256)
        self.encoder4= EncoderBlock(256,512)

        #Latent space
        self.bottleneck= DoubleConv(512, 1024)
        
        #U-Net like decoder with residual connection & spatial attention
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        #Spatial Attention, enhancing the most important regions
        
        #Channel Reductions, no feature extraction
        #self.adjust_enc4 = nn.Conv2d(1280, 640, kernel_size=3, padding=1)
        #self.adjust_enc3 = nn.Conv2d(160, 320, kernel_size=3, padding=1)
        #self.adjust_enc2 = nn.Conv2d(64, 160, kernel_size=3, padding=1)
        self.adjust_out= nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        enc1, x= self.encoder1(x)
        enc2, x= self.encoder2(x)
        enc3, x= self.encoder3(x)
        enc4, x= self.encoder4(x)

        x= self.bottleneck(x)

        x= self.decoder1(x, enc4)
        x= self.decoder2(x, enc3)
        x= self.decoder3(x, enc2)
        x= self.decoder4(x, enc1)

        return self.adjust_out(x)
    
class TabularNet(nn.Module):
    def __init__(self, n_cont_features: int, n_bin_features:int):
        super().__init__()
        self.embed = nn.Embedding(n_bin_features, 64)
        self.proj= nn.Linear(n_cont_features, 64)
        self.attention= nn.MultiheadAttention(
            embed_dim=128, num_heads=4
        )
        self.ffn= nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace= True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace= True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace= True)
        )

    def forward(self, cont: torch.Tensor, bin: torch.Tensor)-> torch.Tensor:
        x= torch.cat([self.proj(cont), self.embed(bin).sum(1)], dim=1)
        x= x.unsqueeze(0)
        x, _= self.attention(x, x, x) 
        return self.ffn(x.squeeze(0))
 

class EfficientUNetWithTabular(nn.Module):
    def __init__(self, n_cont_features, n_bin_features):
        super().__init__()
        self.unet = EfficientUNet()
        self.tabnet = TabularNet(n_cont_features, n_bin_features)
        self.bn= nn.BatchNorm1d(64)
        self.fc= nn.Linear(64, 1)


    def forward(self, img, cont, bin):
        img_features = self.unet(img)        
        tab_features = self.tabnet(cont, bin)

        combined = torch.cat([img_features, tab_features], dim=1)
        return self.fc(self.bn(combined))

if '__main__' == __name__:
    img= torch.randn(32, 3, 224, 224)
    cont= torch.randn(32, 31)
    bin= torch.randint(0, 5, (32, 1))
    
    model = EfficientUNetWithTabular(31, 5)
    output = model(img, cont, bin)
    print(output.squeeze(dim=1))
    print(model, f'\nTrainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):.2e}')