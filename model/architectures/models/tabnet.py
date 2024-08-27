import torch
import torch.nn as nn

__all__= ['TabNet']

class TabNet(nn.Module):
    """
    A neural network module for processing continuous and binary features using 
    an embedding layer, projection layer, multi-head attention, and feedforward network.

    Args:
        n_cont_features (int): Number of continuous features.
        n_bin_features (int): Number of binary features.
        embed_dim (int): Dimension of the embedding space for binary features. Default is 64.
        use_attention (bool): Whether to use multi-head attention. Default is True.
        attention_heads (int): Number of heads in the multi-head attention mechanism. Default is 4.
        hidden_dim (int): Dimension of the hidden layers in the feedforward network. Default is 128.
        min_dim (int): Minimum dimension of the hidden layers in the feedforward network. Default is 64.
        out_dim (int): Dimension of the output layer. Default is 32.
        depth (int): Number of layers in the feedforward network. Default is 1.
        dropout_rate (float): Dropout rate used in the feedforward network. Default is 0.3.
    """
    def __init__(
        self, 
        n_cont_features: int, 
        n_bin_features: int,
        embed_dim: int= 64,
        use_attention: bool = True,
        attention_heads: int= 4,
        hidden_dim: int= 128,
        min_dim: int= 64,
        out_dim: int= 32, 
        depth: int= 1,
        dropout_rate: float= 0.3
    ) -> None:
        super().__init__()
        
        self.embed = nn.Embedding(n_bin_features, embed_dim)
        self.proj = nn.Linear(n_cont_features, embed_dim)
        self.use_attention = use_attention
        
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim * 2, 
                num_heads=attention_heads
            )

        current_dim= hidden_dim

        ffn_layers = [nn.Linear(2*embed_dim, current_dim)]
        for _ in range(depth):
            ffn_layers.extend([
                nn.BatchNorm1d(current_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(current_dim, max(current_dim // 2, min_dim)),
            ])

            current_dim = max(current_dim // 2, min_dim)
            
        ffn_layers.extend([
            nn.BatchNorm1d(current_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(current_dim, out_dim)
        ])
        
        self.ffn = nn.Sequential(*ffn_layers)

    def forward(self, cont: torch.Tensor, bin: torch.Tensor) -> torch.Tensor:
        cont_proj = self.proj(cont)
        bin_embed = self.embed(bin).sum(1)
        x = torch.cat([cont_proj, bin_embed], dim=1)
        
        if self.use_attention:
            x = x.unsqueeze(0)
            x, _ = self.attention(x, x, x)
            x = x.squeeze(0)
        
        return self.ffn(x)