import torch
import timm
import torch.nn as nn

from .models import TabNet
from .base import ISICModel

__all__= ['LightMultiModalNN']

class LightMultiModalNN(nn.Module, ISICModel):
    def __init__(
        self, 
        tabular_hidden_dim: int = 128,
        tabular_out_dim: int = 380,
        tabular_dropout: float = 0.3,
        classifier_hidden_dim: int = 64,
        classifier_dropout: float = 0.2,
    ) -> None:
        
        super().__init__()

        self.model_eva = timm.create_model( 
            'eva02_small_patch14_336.mim_in22k_ft_in1k', pretrained=True
        )

        in_features = self.model_eva.num_features
        self.model_eva.head = nn.Identity()

        self.tabular_net = nn.Sequential(
            nn.Linear(31+ 6, tabular_hidden_dim),
            nn.SiLU(),
            nn.Dropout(tabular_dropout),
            nn.Linear(tabular_hidden_dim, tabular_out_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features + tabular_out_dim, classifier_hidden_dim),
            nn.GELU(),  
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, continuous:torch.Tensor,  binary:torch.Tensor)-> torch.Tensor:
        eva_out= self.model_eva(image)
        tab_out = self.tabular_net(torch.cat([continuous, binary], dim=1))
        combined = torch.cat([eva_out, tab_out], dim=1)

        out = self.classifier(combined)
        return out