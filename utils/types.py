import torch
from torch import Tensor
from typing import  TypeAlias
from jaxtyping import Float, Int, UInt8

Tensor2D:TypeAlias = Float[Tensor, "height width"]
Tensor3D:TypeAlias = Float[Tensor, "channel height width"]
Tensor4D:TypeAlias = Float[Tensor, "in_channel out_channel height width"]
Image:TypeAlias = Float[Tensor, "3 height width"]

NMNIST_Data:TypeAlias = Float[Tensor, "timestep 2 34 34"]
MNIST_DoG_Data:TypeAlias = UInt8[Tensor, "batch timestep 6 28 28"]
MNIST_DoG_Target:TypeAlias = Int[Tensor, "batch"]