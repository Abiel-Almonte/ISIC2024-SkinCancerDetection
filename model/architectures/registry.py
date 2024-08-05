from base import ModelProtocol
from typing import Type, Dict, Tuple, Any

__all__= ['model_registry', 'register_model', 'get_model']

model_registry:  Dict[str, Tuple[Type[ModelProtocol], Dict[str, Any]]] = {}

def register_model(name: str, ModelClass: Type[ModelProtocol], args: Dict[str, Any]):
    model_registry[name] = {ModelClass, args}

def get_model(name: str) -> Type[ModelProtocol]:
    if name not in model_registry:
        raise ValueError(f"Model {name} not found. Choose one of the following: {list(model_registry.keys())}")
    return model_registry[name]