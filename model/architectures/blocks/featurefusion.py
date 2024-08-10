import torch
import torch.nn as nn

__all__= ['FeatureFusion']

class FeatureFusion(nn.Module):
    def __init__(
        self,
        img_features: int,
        tab_features: int,
        use_attention: bool = True,
        attention_heads: int = 4,
        hidden_dim: int = 128,
        min_dim: int= 64,
        out_dim: int = 64,
        depth: int= 0,
        dropout_rate: float = 0.1
    ) -> None:
        """
        Initialize the FeatureFusion module.

        Args:
            img_features (int): Number of features in image data.
            tab_features (int): Number of features in tabular data.
            use_attention (bool): Whether to use multi-head attention. Default is True.
            attention_heads (int): Number of attention heads in MultiheadAttention. Default is 4.
            hidden_dim (int): Dimension of the first hidden layer in the feedforward network. Default is 128.
            min_dim (int): Minimum dimension of the hidden layers in the feedforward network. Default is 64.
            out_dim (int): Dimension of the output hidden layer in the feedforward network. Default is 64.
            depth (int): Number of layers in the feedforward network. Default is 0.
            dropout_rate (float): Dropout rate applied after the first hidden layer. Default is 0.1.
        """
        super().__init__()
        
        total_features= img_features+ tab_features
        self.use_attention= use_attention

        if self.use_attention:
            self.attention= nn.MultiheadAttention(
                embed_dim= total_features,
                num_heads= attention_heads
            )
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(total_features),
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim)
        )

        current_dim= hidden_dim

        ffn_layers = [nn.LayerNorm(total_features), nn.Linear(total_features, current_dim)]
        for _ in range(depth):
            ffn_layers.extend([
                nn.ReLU(inplace= True),
                nn.Dropout(dropout_rate),
                nn.Linear(current_dim, max(current_dim// 2, min_dim)),
            ])

            current_dim = max(current_dim// 2, min_dim)
            
        ffn_layers.extend([
            nn.ReLU(inplace= True),
            nn.Dropout(dropout_rate),
            nn.Linear(current_dim, out_dim)
        ])

        self.ffn = nn.Sequential(*ffn_layers)

    def forward(self, img_feat: torch.Tensor, tab_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([img_feat, tab_feat], dim=1)

        if self.use_attention:
            combined = combined.unsqueeze(0)
            combined, _ = self.attention(combined, combined, combined)
            combined= combined.squeeze(0)

        return self.ffn(combined)
