import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, ViTForImageClassification
import einops

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
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
    

class TransposedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.deconv(x)
        if skip is not None:
            x = x + skip
        return F.relu(x)

class ViTUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #config = ViTConfig()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

        #for param in self.vit.parameters():
           #param.requires_grad= False

        self.vit.classifier= nn.Linear(self.vit.classifier.in_features, 32)

        #self.self_attention = SelfAttention(self.vit.config.hidden_size)
        #self.bottleneck = nn.Sequential(
         #   nn.Conv2d(self.vit.config.hidden_size, 320, kernel_size=3, padding=1),
         #   nn.BatchNorm2d(320),
         #   nn.ReLU(inplace=True),
         #   nn.Conv2d(320, 320, kernel_size=3, padding=1),
         #   nn.BatchNorm2d(320),
         #   nn.ReLU(inplace=True)
        #)
        #self.dropout = nn.Dropout(0.5)

        #self.decoder = nn.ModuleList([
        #    TransposedConvBlock(320, 192),
        #    TransposedConvBlock(192, 112),
        #    TransposedConvBlock(112, 80),
        #    TransposedConvBlock(80, 40),
        #    TransposedConvBlock(40, 32)
        #])

        #self.adjust_out = nn.Sequential(
        #    nn.Conv2d(self.vit.config.hidden_size, 32, kernel_size=3, padding=1),
        #    nn.BatchNorm2d(32),
        #    nn.ReLU(inplace=True),
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten()
        #)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.vit(x).last_hidden_state[:, 1:, :]

        #b, hw, c= x.shape
        #h= w = int((hw)**0.5)
       # x = x.permute(0, 2, 1).view(b, c, h, w)

        #x = self.self_attention(x)
        #x = self.bottleneck(x)
        #x = self.dropout(x)

        #for i, layer in enumerate(self.decoder):
            #x = layer(x)

        return self.vit(x).logits.squeeze(1)


class TabularNet(nn.Module):
    def __init__(self, n_cont_features: int, n_bin_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_cont_features, 64)
        self.embed = nn.Embedding(n_bin_features, 64)
        self.mha = nn.MultiheadAttention(128, 4)
        self.ffn = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
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

    def forward(self, continuous: torch.Tensor, binary: torch.Tensor) -> torch.Tensor:
        continuous, binary = self.proj(continuous), self.embed(binary).sum(1)
        x = torch.cat([continuous, binary], dim=1)
        #x, _ = self.mha(x, x, x)
        return self.ffn(x)

class FeatureFusion(nn.Module):
    def __init__(self, img_features, tab_features):
        super().__init__()
        self.attention = nn.MultiheadAttention(img_features + tab_features, 4)
        self.norm = nn.LayerNorm(img_features + tab_features)
        self.ffn = nn.Sequential(
            nn.Linear(img_features + tab_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, img_feat, tab_feat):
        combined = torch.cat([img_feat, tab_feat], dim=1).unsqueeze(0)
        #attn_out, _ = self.attention(combined, combined, combined)
        norm_out = self.norm(combined)
        return self.ffn(norm_out.squeeze(0))

class ViTWithTabular(nn.Module):
    def __init__(self, n_cont_features: int, n_bin_features: int) -> None:
        super().__init__()
        self.vit_unet = ViTUNet()
        self.tabnet = TabularNet(n_cont_features, n_bin_features)
        self.fusion = FeatureFusion(32, 32)
        self.ffn = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64, 1))

    def forward(self, image: torch.Tensor, continuous: torch.Tensor, binary: torch.Tensor) -> torch.Tensor:
        image_features = self.vit_unet(image)
        tabular_features = self.tabnet(continuous, binary)
        fused_features = self.fusion(image_features, tabular_features)
        return self.ffn(fused_features)
    
if '__main__' == __name__:
    from torchsummary import summary
    img= torch.randn(32, 3, 224, 224)
    cont= torch.randn(32, 31)
    bin= torch.randint(0, 5, (32, 1))
    
    model= ViTWithTabular(31, 5)
    output= model(img, cont, bin)
    summary(model, input_size= [(3, 224, 224), (31,), (5,)],  batch_dim=32, depth=3, verbose=1)