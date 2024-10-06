import torch
from .types import *
from dataclasses import dataclass

@dataclass
class DistInfo:
    min:float = NotImplemented
    max:float = NotImplemented
    mean:float = NotImplemented
    std:float = NotImplemented
    
@dataclass
class ExpInfo:
    use_cuda:bool = True