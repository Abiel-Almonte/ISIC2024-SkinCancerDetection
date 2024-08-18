import torch
import timm
import torch.nn as nn

from .models import TabNet
from .base import ISICModel

__all__= ['EfficientNetEVAModel']

class EfficientNetEVAModel(nn.Module, ISICModel):
    def __init__(self):
        super().__init__()

        self.model_eva = timm.create_model( 
            'eva02_small_patch14_336.mim_in22k_ft_in1k', pretrained=True
        )

        in_features = self.model_eva.num_features
        self.model_eva.head = nn.Identity()

        self.tabular_net = nn.Sequential(
            nn.Linear(31+ 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 380),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features + 380, 64),
            nn.GELU(),  
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, continuous:torch.Tensor,  binary:torch.Tensor)-> torch.Tensor:
        eva_out= self.model_eva(image)
        tab_out = self.tabular_net(torch.cat([continuous, binary], dim=1))
        combined = torch.cat([eva_out, tab_out], dim=1)


        out = self.classifier(combined)
        return out