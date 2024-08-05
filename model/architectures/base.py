import torch
from typing import Protocol, Any

__all__= ['ModelProtocol']

class ModelProtocol(Protocol):
    def forward(
        self,
        image:torch.Tensor,
        continous:torch.Tensor, 
        binary:torch.Tensor
    ) -> Any: ...
