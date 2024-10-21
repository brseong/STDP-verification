import logging
from typing import Any
from torchvision import utils
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import _axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .types import Tensor2D, Tensor4D, Image

def conv_weight2img(tensor:Tensor4D, ch:int=0, allkernels:bool=False, nrow:int=8, padding:int=1) -> Image:
    assert tensor.shape[1] == 3 # The number of channels must be 3.
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    # return grid
    assert len(grid.shape) == 3
    return grid.permute(1,2,0)

__axe2caxe = {}
def draw_weight_map(fig:Figure, axe:_axes.Axes, weight_tensor:Image):
    pc = axe.imshow(weight_tensor) # type: ignore
    
    if id(axe) not in __axe2caxe:
        div = make_axes_locatable(axe) # get division of the subplot.
        __axe2caxe[id(axe)] = div.append_axes("right", size="5%", pad=0.05) # type: ignore # new ax for color bar
    fig.colorbar(pc, __axe2caxe[id(axe)]) # type: ignore

def log(msg:Any):
    print(msg)
    logging.getLogger().info(msg)