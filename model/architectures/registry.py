from .base import ISICModel
from typing import Type, Dict, Tuple, Any

__all__= ['model_registry', 'register_model', 'get_model']

model_registry:  Dict[str, Tuple[Type[ISICModel], Dict[str, Any]]] = {}

def register_model(name: str, ModelClass: Type[ISICModel], args: Dict[str, Any]):
    model_registry[name] = (ModelClass, args)

def get_model(name: str) -> Type[ISICModel]:
    if name not in model_registry:
        raise ValueError(f"Model {name} not found. Choose one of the following: {list(model_registry.keys())}")
    return model_registry[name]