import os
import random
import numpy
import torch
import yaml
from .pipeline import train_evaluate_model, test_model
from typing import Callable, Any, Dict

__all__= ['train_evaluate_test', 'cuda_stream_wrapper', 'load_config', 'set_seed']

def train_evaluate_test(
    train_evaluate_args: Dict[Any, Any], 
    test_args: Dict[Any, Any]
) -> Any:
    """
    Executes training and evaluation, and optionally tests the model.
    
    Parameters:
        train_evaluate_args (Dict[Any, Any]): Arguments for training and evaluation.
        test_args (Dict[Any, Any]): Arguments for testing the model.

    Returns:
        Any: The result of the test model function if test_args is provided.
    """
    run = cuda_stream_wrapper(train_evaluate_model, train_evaluate_args)
    if test_args is not None:
        test_args['config']['testing'].update({'run': run})
        return cuda_stream_wrapper(test_model, test_args)

def cuda_stream_wrapper(
    fn: Callable[[Any], Any], 
    kwargs: Dict[str, Any]
) -> Any:
    """
    Executes a function within a CUDA stream and manages CUDA resources.
    
    Parameters:
        fn (Callable[[Any], Any]): The function to execute.
        kwargs (Dict[str, Any]): Keyword arguments to pass to the function.

    Returns:
        Any: The result of the function execution.
    """
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        result = fn(**kwargs)

    torch.cuda.current_stream().wait_stream(stream)
    del stream
    torch.cuda.empty_cache()
    
    return result

def load_config(fp: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    
    Parameters:
        fp (str): The file path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration.
    """
    with open(fp, 'r') as file:
        return yaml.safe_load(file)

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility.
    
    Parameters:
        seed (int): The seed value to use.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
