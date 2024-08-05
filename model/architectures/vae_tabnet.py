import torch
import torch.nn as nn
from .models import VAE, TabNet
from .base import ISICModel
from typing import Tuple

__all__= ['VAETabNet']

class VAETabNet(nn.Module, ISICModel):
    def __init__(self, cont_features:int= 31, bin_features: int= 6, latent_dim: int= 256) -> None:
        super().__init__()
        self.tabnet= TabNet(cont_features, bin_features)
        self.imgnet= VAE(latent_dim)
        self.ffn= nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace= True),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
        )
    def forward(self, image:torch.Tensor, cont:torch.Tensor, bin:torch.Tensor)->Tuple[torch.Tensor, ...]:
        image_features, mu, logvar= self.imgnet(image)
        tabular_features= self.tabnet(cont, bin)
        combined= torch.cat([image_features, tabular_features], dim=1)
        out=self.ffn(combined)
        return out, mu, logvar

if '__main__' == __name__:
    from torchsummary import summary
    img= torch.randn(32, 3, 448, 448)
    cont= torch.randn(32, 31)
    bin= torch.randint(0, 5, (32, 1))
    
    model= VAETabNet(31, 5)
    output= model(img, cont, bin)
    summary(model, input_size= [(3, 224, 224), (31,), (5,)],  batch_dim=32, depth=3, verbose=1)