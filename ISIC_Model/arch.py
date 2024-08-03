import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import einops
    
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
    def __init__(self, in_channels:int, r:int= 8) -> None:
        super().__init__()
        self.channel_attention= ChannelAttention(in_channels, r)
        self.spatial_attention= SpatialAttention()

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x= self.channel_attention(x)
        x= self.spatial_attention(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels: int)-> None:
        super().__init__()
        self.query= nn.Conv2d(in_channels, in_channels//8, kernel_size=3, padding=1)
        self.key= nn.Conv2d(in_channels, in_channels//8, kernel_size=3, padding=1)
        self.value= nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gamma= nn.Parameter(torch.zeros(1))
        self.scale= (in_channels//8)** -0.5
    
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        _, _, h, w= x.shape
        q= einops.rearrange(self.query(x), 'b c h w -> b (h w) c')
        k= einops.rearrange(self.key(x), 'b c h w -> b c (h w)')
        v= einops.rearrange(self.value(x), 'b c h w -> b c (h w)')

        attn= F.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out= einops.rearrange(torch.bmm(v, attn.permute(0,2,1)), 'b c (h w) -> b c h w', h= h, w= w)
        return self.gamma* out+ x
    
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
        self.attention1= CBAM(640)
        self.attention2= CBAM(320)
        self.attention3= CBAM(160)
        self.attention4= CBAM(64)
        
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
    from torchsummary import summary
    img= torch.randn(32, 3, 224, 224)
    cont= torch.randn(32, 31)
    bin= torch.randint(0, 5, (32, 1))
    
    model= EfficientUNetWithTabular(31, 5)
    model(img, cont, bin)
    summary(model, input_size= [(3, 224, 224), (31,), (5,)],  batch_dim=32, depth=3, verbose=1)