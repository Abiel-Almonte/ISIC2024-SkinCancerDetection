import torch
import torch.nn as nn
from .blocks import CrossModalAttention
from .base import ISICModel
from torchvision.models import vit_b_16

__all__= ['CrossModalTransformer']

class CrossModalTransformer(nn.Module, ISICModel):
    def __init__(self, cont_features: int= 31, bin_features: int= 6, image_size:int= 224, dim: int= 768, depth: int= 4, mlp_dim: int= 3072):
        """
        Initialize the CrossModalTransformer model.

        Args:
            cont_features (int): Number of continuous features.
            bin_features (int): Number of binary features.
            image_size (int): Resolution of input image.
            dim (int): Dimension of the transformer model.
            depth (int): Number of transformer layers.
            mlp_dim (int): Dimension of the feed-forward network in the transformer.
        """
        super().__init__()
        self.vit = vit_b_16(image_size= image_size)
        self.vit.heads = nn.Identity()
        self.vit_out= nn.Linear(self.vit.hidden_dim, dim//2)
        self.cont_embed = nn.Linear(cont_features, dim//4)
        self.bin_embed= nn.Embedding(bin_features, dim//4)
        
        self.transformer = nn.ModuleList([
            nn.ModuleList([
                CrossModalAttention(dim),
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, dim),
                nn.LayerNorm(dim)
            ]) for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, image:torch.Tensor, continous:torch.Tensor, binary:torch.Tensor):
        image= self.vit(image)
        image= self.vit_out(image)
        tabular= torch.cat([self.cont_embed(continous), self.bin_embed(binary).sum(1)], dim=1)
        x = torch.cat([image, tabular], dim=1)
        for attn, norm1, ff1, gelu, ff2, norm2 in self.transformer:
            attn_out = attn(x, x)
            x = norm1(x + attn_out)
            ff_out = ff2(gelu(ff1(x)))
            x = norm2(x + ff_out)

        return self.mlp_head(x)

if '__main__' == __name__:
    from torchsummary import summary
    img= torch.randn(32, 3, 224, 224)
    cont= torch.randn(32, 31)
    bin= torch.randint(0, 6, (32, 1))
    
    model= CrossModalTransformer()
    output= model(img, cont, bin)
    summary(model, input_size= [(3, 224, 224), (31,), (5,)],  batch_dim=32, depth=3, verbose=1)