from torchvision import utils
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import _axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .types import Tensor2D

def tensor2img(tensor:torch.Tensor, ch=0, allkernels=False, nrow=8, padding=1) -> torch.Tensor:
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    return grid.permute(1,2,0)

_axe2caxe = {}
def draw_weight_map(fig:Figure, axe:_axes.Axes, weight_tensor:Tensor2D):
    pc = axe.pcolor(weight_tensor)
    
    if id(axe) not in _axe2caxe:
        div = make_axes_locatable(axe) # get division of the subplot.
        _axe2caxe[id(axe)] = div.append_axes("right", size="5%", pad=0.05) # new ax for color bar
    fig.colorbar(pc, _axe2caxe[id(axe)])

def draw_tensors(tensor:torch.Tensor|list[torch.Tensor], ch=0, allkernels=False, nrow=8, padding=1) -> None:
    if not isinstance(tensor, list):
        tensor = [tensor]
    
    axes:list[_axes.Axes]
    fig, axes = plt.subplots(len(tensor), 1)
    fig.set_size_inches(nrow,cols := np.min((tensor.shape[0] // nrow + 1, 64)))
    for ax, values in zip(axes, tensor):
        ax.imshow(tensor2img(values, ch=ch, allkernels=allkernels, nrow=nrow, padding=padding).numpy())
    
    