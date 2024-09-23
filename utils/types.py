import torch
from typing import NewType, Annotated
from torchtyping import patch_typeguard
from torchtyping import TensorType as Tensor

patch_typeguard()

Tensor2D = Tensor["height", "width", float]

# for N-MNIST
NMNIST = Tensor["timesteps", 2, 34, 34]