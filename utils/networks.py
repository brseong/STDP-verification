import torch
from torch import nn
from typeguard import typechecked
from typing import cast
from abc import abstractmethod, ABCMeta
from .types import Tensor2D
from .dataclasses import DistInfo
from .spikingjelly.spikingjelly.activation_based import layer, learning, neuron

class STDPNet(nn.Module, metaclass=ABCMeta):
    learners:list[learning.STDPLearner] = NotImplemented
    draw_ids:tuple[int, ...] = NotImplemented
        
    # def gen_block(self, in_features:int, out_features:int, **kwargs) -> nn.Sequential:
    #     return nn.Sequential(
    #         layer.Linear(in_features, out_features, **kwargs),
    #         neuron.LIFNode()
    #     )

    @abstractmethod
    def post_optim(self) -> None: pass
    
    @abstractmethod
    def draw_weights(self, id:int=0) -> Tensor2D: pass
    
    @staticmethod
    @abstractmethod
    def f_weight(x) -> torch.Tensor: pass
    
    @staticmethod
    @abstractmethod
    def f_pre(x, w_min, alpha=0.) -> float: pass

    @staticmethod
    @abstractmethod
    def f_post(x, w_max, alpha=0.) -> float: pass
    
    
class DiehlAndCook2015(STDPNet):
    draw_ids = (0,1)
    def __init__(
        self, in_features:int,
        hidden_features:int,
        tau_pre:float,
        tau_post:float,
        w_info:DistInfo,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_features, self.hidden_features = in_features, hidden_features
        self.w_info = w_info
        
        self.excitatory = layer.Linear(in_features, hidden_features, bias=False)
        self.inhibitory = layer.Linear(hidden_features, hidden_features, bias=False)
        self.inhibitory.weight.data = w_info.mean*(-1+torch.eye(hidden_features))
        self.lif_hidden = neuron.LIFNode()
        
        # Init using truncated normal distribution
        nn.init.trunc_normal_(self.excitatory.weight.data, w_info.mean, w_info.std, w_info.min, w_info.max)
        
        # Initialize STDP Learners
        self.learners:list[learning.STDPLearner] = []
        # STDP_TARGETS = [self.excitatory, self.inhibitory]
        STDP_TARGETS = [self.excitatory]
        
        for linear in STDP_TARGETS:
            self.learners.append(
                learning.STDPLearner(step_mode='s', synapse=linear, sn=self.lif_hidden, 
                                        tau_pre=tau_pre, tau_post=tau_post,
                                        f_pre=DiehlAndCook2015.f_weight, f_post=DiehlAndCook2015.f_weight)
            )
    
    ### TODO: In order to balance the network minimizing the risk of neurons totally dominating the output, neurons should have approximately the same firing rates.
    ### This can be achieved by increasing the threshold for a neuron once it fires and have it slowly decay to some resting value over time.
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x[0].reshape(-1).shape[0] == self.in_features, f"{x[0].reshape(-1).shape[0]}, {self.in_features}"
        excitatory = self.lif_hidden(self.excitatory(x)) # excitation from input
        inhibitory = self.lif_hidden(self.inhibitory(excitatory)) # lateral inhibition
        return excitatory
    
    def post_optim(self) -> None:
        self.excitatory.weight.data.clamp_(self.w_info.min, self.w_info.max)
        self.inhibitory.weight.data.clamp_(-self.w_info.max, -self.w_info.min)
    
    @staticmethod
    def f_weight(x) -> torch.Tensor:
        return torch.clamp(x, -1, 1.)
    
    @staticmethod
    def f_pre(x, w_min, alpha=0.) -> float:
        return (x - w_min) ** alpha
    
    @staticmethod
    def f_post(x, w_max, alpha=0.) -> float:
        return (w_max - x) ** alpha
    
    @typechecked
    def draw_weights(self, id:int=0) -> Tensor2D:
        assert id in self.draw_ids
        target = [self.excitatory, self.inhibitory][id]
        block_size:tuple[int,int] = [(2*34,34), (10,1)][id]
        
        weight = target.weight.data.cpu().detach()
        canvas = cast(Tensor2D, torch.zeros((block_size[0], block_size[1]*self.hidden_features)))
        for i_neuron in range(self.hidden_features):
            canvas[:, block_size[1]*i_neuron:block_size[1]*(i_neuron+1)] = weight[i_neuron].reshape(*block_size)
        return canvas