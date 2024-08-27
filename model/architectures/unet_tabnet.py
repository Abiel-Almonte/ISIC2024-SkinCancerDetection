import torch
import torch.nn as nn
from .models import TabNet, EfficientUNet, EfficientUNet_v2, ViTUNet
from .blocks import FeatureFusion
from .base import ISICModel

__all__= ['EfficientUNetWithTabular', 'EfficientUNetWithTabular_v2', 'ViTWithTabular']

class EfficientUNetWithTabular(nn.Module, ISICModel):
    def __init__(self, cont_features:int= 31, bin_features: int= 6):
        """
        Initialize the EfficientUNetWithTabular model.

        Args:
            cont_features (int): Number of continuous features.
            bin_features (int): Number of binary features.
        """
        super().__init__()
        self.unet = EfficientUNet()
        self.tabnet = TabNet(cont_features, bin_features)
        self.fc = nn.Linear(64, 1)


    def forward(self, image:torch.Tensor, continous:torch.Tensor, binary:torch.Tensor):
        img_features = self.unet(image)        
        tab_features = self.tabnet(continous, binary)

        combined = torch.cat([img_features, tab_features], dim=1)
        return self.fc(combined)

class EfficientUNetWithTabular_v2(nn.Module,  ISICModel):
    def __init__(self, cont_features:int= 31, bin_features:int= 6) -> None:
        """
        Initialize the EfficientUNetWithTabular_v2 model.

        Args:
            cont_features (int): Number of continuous features.
            bin_features (int): Number of binary features.
        """
        super().__init__()
        self.unet= EfficientUNet_v2()
        self.tabnet= TabNet(cont_features, bin_features, use_attention=False, out_dim=16, hidden_dim=512, depth=2)
        self.fusion= FeatureFusion(img_features= 32, tab_features= 16, out_dim= 32, use_attention=False)
        self.ffn= nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Linear(32, 1))

    def forward(self, image:torch.Tensor, continous:torch.Tensor, binary:torch.Tensor)-> torch.Tensor:
        image_features= self.unet(image)
        tabular_features= self.tabnet(continous, binary)
        fused_features = self.fusion(image_features, tabular_features)
        return self.ffn(fused_features)


class ViTWithTabular(nn.Module, ISICModel):
    def __init__(self, cont_features: int, bin_features: int) -> None:
        """
        Initialize the ViTWithTabular_v2 model.

        Args:
            cont_features (int): Number of continuous features.
            bin_features (int): Number of binary features.
        """
        super().__init__()
        self.vit_unet = ViTUNet()
        self.tabnet = TabNet(cont_features, bin_features, use_attention=False, out_dim= 16, hidden_dim=512, depth=2)
        self.fusion = FeatureFusion(img_features=32,  tab_features=16, out_dim=48)
        self.ffn = nn.Sequential(
            nn.BatchNorm1d(48),
            nn.Linear(48, 1))

    def forward(self, image: torch.Tensor, continuous: torch.Tensor, binary: torch.Tensor) -> torch.Tensor:
        image_features = self.vit_unet(image)
        tabular_features = self.tabnet(continuous, binary)
        fused_features = self.fusion(image_features, tabular_features)
        return self.ffn(fused_features)
    
if '__main__' == __name__:
    from torchsummary import summary
    img= torch.randn(32, 3, 448, 448)
    cont= torch.randn(32, 31)
    bin= torch.randint(0, 5, (32, 1))
    
    model= ViTWithTabular(31, 5)
    output= model(img, cont, bin)
    summary(model, input_size= [(3, 224, 224), (31,), (5,)],  batch_dim=32, depth=3, verbose=1)