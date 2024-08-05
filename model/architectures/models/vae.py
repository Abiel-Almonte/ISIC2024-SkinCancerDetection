import torch
import torch.nn as nn
from typing import Tuple

__all__= ['VAE']

class VAE(nn.Module):
    def __init__(self, latent_dim: int= 256) -> None:
        super().__init__()
        self.latent_dim= latent_dim
        self.encoder = nn.Sequential(
            self._conv_block(3, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
        )
        self.fc_mu= nn.Linear(512*5*5, self.latent_dim)
        self.fc_logvar= nn.Linear(512*5*5, self.latent_dim)
        self.decoder_in= nn.Linear(self.latent_dim, 512*5*5)

        self.decoder = nn.Sequential(
            self._deconv_block(512, 256),
            self._deconv_block(256, 128),
            self._deconv_block(128, 64),
            self._deconv_block(64, 32),
        )
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

    def _conv_block(self, in_channels, out_channels)-> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(3, 3)
        )

    def _deconv_block(self, in_channels, out_channels)-> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
        )
    
    def encode(self, x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        x= self.encoder(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)
    
    def reparamaterize(self, mu:torch.Tensor, logvar: torch.Tensor)-> torch.Tensor:
        std= torch.exp(0.5* logvar)
        eps= torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, x:torch.Tensor)-> torch.Tensor:
        x= self.decoder_in(x)
        x= x.view(-1, 512, 5, 5)
        return self.decoder(x)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        mu, logvar= self.encode(x)
        x= self.reparamaterize(mu, logvar)
        x= self.decode(x)
        return self.out(x), mu, logvar