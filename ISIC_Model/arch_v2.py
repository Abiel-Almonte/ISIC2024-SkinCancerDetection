import torch, einops
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SelfAttention(nn.Module):
    def __init__(self, in_channels:int) -> None:
        super().__init__()
        self.query= nn.Conv2d(in_channels, in_channels// 8, kernel_size= 1)
        self.key= nn.Conv2d(in_channels, in_channels// 8, kernel_size= 1)
        self.value= nn.Conv2d(in_channels, in_channels, kernel_size= 1)
        self.gamma= nn.Parameter(torch.zeros(1))

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        _, _, h, w= x.shape
        q= einops.rearrange(self.query(x), 'b c h w -> b (h w) c')
        k= einops.rearrange(self.key(x), 'b c h w -> b c (h w)')
        v= einops.rearrange(self.value(x), 'b c h w -> b c (h w)')

        attn= F.softmax(torch.bmm(q, k), dim=-1)
        attn= einops.rearrange(attn, 'b hw1 hw2 -> b hw2 hw1')
        out= einops.rearrange(torch.bmm(v, attn), 'b c (h w) -> b c h w', h= h, w= w)
        return self.gamma* out+ x
    
class FPNBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        self.adjust= nn.Upsample(scale_factor= 2, mode= 'nearest')
        self.conv= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor, skip: torch.Tensor= None)-> torch.Tensor:
        #skip= self.adjust(x)
        if skip is not None:
            if x.shape != skip.shape:
                x= self.adjust(x)
            x= x+ skip
        return F.relu(self.conv(x))
    
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
            x= layer(x, skips[len(self.decoder)- 1- i])

        return self.adjust_out(x)

class TabularNet(nn.Module):
    def __init__(self, n_cont_features:int, n_bin_features:int) -> None:
        super().__init__()
        self.proj= nn.Linear(n_cont_features, 64)
        self.embed= nn.Embedding(n_bin_features, 64)
        self.mha= nn.MultiheadAttention(128, 4)
        self.ffn= nn.Sequential(
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True)
        )

    def forward(self, continous:torch.Tensor, binary:torch.Tensor)->torch.Tensor:
        continous, binary= self.proj(continous), self.embed(binary).sum(1)
        x= torch.cat([continous, binary], dim=1).unsqueeze(0)
        x, _= self.mha(x, x, x)
        return self.ffn(x.squeeze(0))

class EfficientUNetWithTabular_v2(nn.Module):
    def __init__(self, n_cont_features:int , n_bin_features:int) -> None:
        super().__init__()
        self.unet= EfficientUNet_v2()
        self.tabnet= TabularNet(n_cont_features, n_bin_features)
        self.ffn= nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64,1))

    def forward(self, image:torch.Tensor, continous:torch.Tensor, binary:torch.Tensor)-> torch.Tensor:
        image_features= self.unet(image)
        tabular_features= self.tabnet(continous, binary)
        x= torch.cat([image_features, tabular_features], dim=1)
        return self.ffn(x)
    
if '__main__' == __name__:
    from torchsummary import summary
    img= torch.randn(32, 3, 224, 224)
    cont= torch.randn(32, 31)
    bin= torch.randint(0, 5, (32, 1))
    
    model= EfficientUNetWithTabular_v2(31, 5)
    model(img, cont, bin)
    summary(model, input_size= [(3, 224, 224), (31,), (5,)],  batch_dim=32, depth=3, verbose=1)