import torch

from torch import nn
from torch.nn import Parameter
from torchvision import transforms
import matplotlib.pyplot as plt

from .spikingjelly.spikingjelly.activation_based import layer, learning, neuron
from .spikingjelly.spikingjelly.activation_based.surrogate import Sigmoid

from .SpykeTorch.SpykeTorch import snn, utils
from .SpykeTorch.SpykeTorch import functional as sf

from typeguard import typechecked
from typing import Callable, cast
from abc import abstractmethod, ABCMeta
from torchtyping import TensorType as Tensor

from .types import Tensor2D, Tensor4D, Image
from .dataclasses import DistInfo
from .visual import conv_weight2img


class LIFNeuron(neuron.SimpleLIFNode):
    def __init__(self, tau: float, decay_input: bool, v_threshold: float = 1,
                 v_reset: float = 0, surrogate_function: Callable = neuron.surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s'):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
    def neuronal_charge(self, x:torch.Tensor):
        return super().neuronal_charge(x)

class STDPNet(nn.Module, metaclass=ABCMeta):
    learners:list[learning.STDPLearner] = NotImplemented
    draw_ids:tuple[int, ...] = NotImplemented
    
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
    
class Mozafari2018(STDPNet):
    draw_ids = (0, 1, 2)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) # type: ignore
        
        self.conv1 = snn.Convolution(6, 30, 5, 0.8, 0.05)
        self.conv1_t = 15
        self.k1 = 5
        self.r1 = 3

        self.conv2 = snn.Convolution(30, 240, 3, 0.8, 0.05)
        self.conv2_t = 10
        self.k2 = 8
        self.r2 = 1

        self.conv3 = snn.Convolution(240, 200, 5, 0.8, 0.05)

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.decision_map:list[int] = []
        for i in range(10):
            self.decision_map.extend([i]*20)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0
        
    def forward(self, input:Tensor["batch", "timesteps", "channels1", "height1", "width1"], max_layer:int):
        _input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            pot:Tensor["batch", "timesteps", "channels2", "height2", "width2"]
            pot = self.conv1(_input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.ctx["input_spikes"] = _input
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                self.ctx["input_spikes"] = spk_in
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2))
            pot = self.conv3(spk_in)
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        else:
            pot = self.conv1(_input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                return spk, pot
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                return spk, pot
            pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
    
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
    
    def post_optim(self) -> None:
        pass
    
    def draw_weights(self, id:int=0) -> Image:
        if id in self.draw_ids:
            conv = [self.conv1, self.conv2, self.conv3][id]
            nrow = [12, 50, 160][id]
        else:
            raise ValueError("Weight id is not valid.")
        # Get first 3 convolution kernels' weights.
        weight:Tensor4D = conv.weight.detach().cpu().clone()
        assert len(weight.shape)==4
        weight = weight[:,:3]
        return conv_weight2img(weight, nrow=nrow, padding=1)
    
    @staticmethod
    def generate_transform():
        """Applies difference of gaussian filters and temporal encoding. 6 DoG kernels are applied to image data, and returns data with 6 channels and 15 time steps.
        E.g. With MNIST, Image is converted to (15, 6, 28, 28) Tensor.

        Returns:
            _type_: S1C1Transform object
        """
        class S1C1Transform:
            def __init__(self, filter:utils.Filter, timesteps:int = 15):
                self.to_tensor = transforms.ToTensor()
                self.filter = filter
                self.temporal_transform = utils.Intensity2Latency(timesteps)
                self.cnt = 0
            def __call__(self, image:torch.Tensor):
                if self.cnt % 1000 == 0:
                    print(self.cnt)
                self.cnt+=1
                image = self.to_tensor(image) * 255
                image.unsqueeze_(0)
                image = self.filter(image)
                image = sf.local_normalization(image, 8)
                temporal_image = self.temporal_transform(image)
                return temporal_image.sign().byte()

        kernels = [ utils.DoGKernel(3,3/9,6/9),
                    utils.DoGKernel(3,6/9,3/9),
                    utils.DoGKernel(7,7/9,14/9),
                    utils.DoGKernel(7,14/9,7/9),
                    utils.DoGKernel(13,13/9,26/9),
                    utils.DoGKernel(13,26/9,13/9)]
        filter = utils.Filter(kernels, padding = 6, thresholds = 50)
        return S1C1Transform(filter)
    
    @staticmethod
    def f_weight(x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def f_pre(x:torch.Tensor, w_min:float, alpha:float=0.) -> float:
        raise NotImplementedError

    @staticmethod
    def f_post(x:torch.Tensor, w_max:float, alpha:float=0.) -> float:
        raise NotImplementedError

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