import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
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

class SpatialAttention(nn.Module):
    def __init__(self, in_channels)-> None:
        super().__init__()
        #Create attention map and obtain scores
        self.attention= nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size= 1),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return x* self.attention(x)
        
class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_skip: bool= False)-> None:
        super().__init__()
        self.use_skip= use_skip
        if use_skip:
            self.upsample= nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.channel_reduction= nn.Conv2d(in_channels, out_channels, kernel_size= 1)
        self.double_conv= DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace= True)
        self.convT = nn.ConvTranspose2d(out_channels, out_channels, 2, stride= 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor= None)-> torch.Tensor:
        if self.use_skip:
            x= x+ self.upsample(skip)
        
        res= self.channel_reduction(x)
        x= self.double_conv(x)
        x= self.relu(x + res)
        return self.convT(x)
    
class EfficientUNet(nn.Module):
    def __init__(self)-> None:
        super().__init__()
        
        #EfficientNet encoder
        self.encoder = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        ).features
        
        #U-Net like decoder with residual connection
        self.decoder1 = ResidualDecoderBlock(1280, 640)
        self.decoder2 = ResidualDecoderBlock(640, 320, use_skip= True)
        self.decoder3 = ResidualDecoderBlock(320, 160, use_skip= True)
        self.decoder4 = ResidualDecoderBlock(160, 64, use_skip= True)

        #Spatial Attention, enhancing the most important regions
        self.attention1= SpatialAttention(640)
        self.attention2= SpatialAttention(320)
        self.attention3= SpatialAttention(160)
        self.attention4= SpatialAttention(64)
        
        #Channel Reductions, no feature extraction
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
        self.fc = nn.Linear(64, 1)


    def forward(self, img, cont, cat):
        img_features = self.unet(img)        
        tab_features = self.tabnet(cont, cat)

        combined = torch.cat([img_features, tab_features], dim=1)
        return self.fc(combined)

if '__main__' == __name__:
    img = torch.randn(32, 3, 224, 224)
    cont = torch.randn(32, 10)
    cat = torch.randint(0, 5, (32, 1))
    
    model = EfficientUNetWithTabular(10, 5)
    output = model(img, cont, cat)
    print(output.squeeze(dim=1))
    print(model, f'\nTrainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):.2e}')