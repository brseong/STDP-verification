import torch
from typing import NewType, Annotated
from torchtyping import TensorType as Tensor

Tensor2D = Tensor["height", "width", float]