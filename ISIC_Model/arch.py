import torch
import torch.nn as nn
from torchvision import models

class FTTransformer(nn.Module):
    def __init__(self, n_cont_features:int, n_bin_features:int):
        super().__init__()

        self.cont_projection = nn.Linear(n_cont_features, 256)
        self.bin_embedding= nn.Embedding(n_bin_features, 256)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model= 512, nhead= 8), num_layers= 6)
        
        self.ffn= nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64)
        )

    def forward(self, cont_features: torch.Tensor, bin_features:torch.Tensor):
        #normalize
        cont_features= (cont_features - cont_features.mean(dim=0)) / (cont_features.std(dim=0) + 1e-6)

        x= torch.cat([self.cont_projection(cont_features), self.bin_embedding(bin_features).sum(dim=1)], dim=1)
        x= self.transformer_encoder(x.unsqueeze(1))
        return self.ffn(x.squeeze(1))

class doubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int)->None:
        super().__init__()

        self.conv1= nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1= nn.BatchNorm2d(out_channels)
        self.conv2= nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2= nn.BatchNorm2d(out_channels)
        self.relu= nn.ReLU()

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x= self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))
    
class decoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int)->None:
        super().__init__()

        self.convT= nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 2, stride=2, padding=0)
        self.conv= doubleConv(2*out_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x:torch.Tensor, skip:torch.Tensor)->torch.Tensor:
        x= self.convT(x)
        if x.shape != skip.shape:
            skip = self.upsample(skip)
        return self.conv(torch.cat([x, skip], axis=1))
    
class ResUNet(nn.Module):
    def __init__(self, is_with_tabular:bool= False) -> None:
        super().__init__()

        self.is_with_tabular= is_with_tabular

        #backbone
        self.encoder= models.resnet18(weights= models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder= nn.Sequential(
            *list(self.encoder.children())[:-2]
        )

        self.bottleneck= doubleConv(512, 1024)

        self.dec1= decoderBlock(1024, 512)
        self.dec2= decoderBlock(512, 256)
        self.dec3= decoderBlock(256, 128)
        self.dec4= decoderBlock(128, 64)

        self.flatten= nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.ffn= nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        e1= self.encoder[:4](x)
        e2= self.encoder[5](e1)
        e3= self.encoder[6](e2)
        e4= self.encoder[7](e3)

        b= self.bottleneck(e4)

        d1= self.dec1(b, e4)
        d2= self.dec2(d1, e3)
        d3= self.dec3(d2, e2)
        d4= self.dec4(d3, e1)

        if self.is_with_tabular:
            return self.flatten(d4)
        
        return self.ffn(self.flatten(d4))
    
class ResUNetWithTabular(nn.Module):
    def __init__(self, n_cont_features:int, n_bin_features:int)->None:
        super().__init__()
        self.fttransformer= FTTransformer(n_cont_features, n_bin_features)
        self.resunet= ResUNet(True)

        self.ffn= nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, image: torch.Tensor, tabular_cont: torch.Tensor, tabular_bin:torch.Tensor)->torch.Tensor:
        image_features= self.resunet(image)
        tabular_features= self.fttransformer(tabular_cont, tabular_bin)

        x= torch.cat([image_features, tabular_features], dim=1)
        return self.ffn(x)

if '__main__' == __name__:
    model= ResUNetWithTabular(32, 6).to('cuda')
    print(model, f'\nTrainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):.2e}')