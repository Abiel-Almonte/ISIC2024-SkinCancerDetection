import torch
from typing import Protocol, Any, runtime_checkable

__all__= ['ISICModel']

@runtime_checkable
class ISICModel(Protocol):
    def forward(
        self,
        image:torch.Tensor,
        continous:torch.Tensor, 
        binary:torch.Tensor
    ) -> Any: ...
