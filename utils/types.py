import torch
from torch import Tensor
from typing import  Annotated, TypeAlias
from jaxtyping import Float, Int, UInt8

Tensor2D:TypeAlias = Annotated[Tensor, "height width"]
Tensor3D:TypeAlias = Annotated[Tensor, "channel height width"]
Tensor4D:TypeAlias = Annotated[Tensor, "* out_channel height width"]
TensorHidden:TypeAlias = Annotated[Tensor, "timestep channel height width"]
Image:TypeAlias = Annotated[Tensor, "3 height width"]

NMNIST_Data:TypeAlias = Annotated[Tensor, "timestep 2 34 34", torch.float]
MNIST_DoG_Data:TypeAlias = Annotated[Tensor, "batch timestep 6 28 28", torch.uint8]
MNIST_DoG_Target:TypeAlias = Annotated[Tensor, "batch", torch.int]