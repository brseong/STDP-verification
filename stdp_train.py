from typing import NamedTuple
import torch
import torchvision
from random import shuffle
from torch import Tensor
from torch.nn import Threshold, Parameter
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from jaxtyping import Float, Int, UInt8, Bool
from utils.SpykeTorch.SpykeTorch.utils import Intensity2Latency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_steps = 15
batch_size = 1
shuffle = False
num_workers = 4

type BTE = UInt8[Tensor, "Batch Timestep Embedding"]
type BE = UInt8[Tensor, "Batch Embedding"]


class Layer(NamedTuple):
    """Class to store the parameters of a layer.

    Args:
        w: (Parameter) The weights of the layer. Shape: (n_pre, n_post)
        d_thr: (Parameter) The threshold bias of the layer. Shape: (n_post)
        p_t: (Tensor) The potential of the layer. Shape: (n_post)
        e_t: (Tensor) The eligibility trace of the layer. Shape: (n_pre, n_post)
    """

    w: Parameter
    d_thr: Parameter
    p_t: Tensor
    e_t: Tensor


def to_synapsewise_epsilon_greedy(x: Tensor, epsilon: float = 1e-1):
    """Make x to actionwise epsilon-greedy. Assumes and treats each element of x as an binary action distribution about spike or not spike,
    and applies epsilon-greedy to each action distribution.
    e.g. [0, 1, 0.5] -> [\epsilon/2, (1-\epsilon)*1+\epsilon/2, (1-\epsilon)*0.5+\epsilon/2]

    Args:
        x (Tensor): Original action distribution. Shape: (Batch, n_actions)
        epsilon (float, optional): Probability to take random action. Defaults to 1e-1.

    Returns:
        Tensor: Epsilon-greedy action distribution. Shape: (Batch, n_actions)
    """
    return x * (1 - epsilon) + epsilon / 2


class STDPNet(torch.nn.Module):
    def __init__(
        self,
        n_features: tuple[int, ...] = (784, 128, 10),
    ):
        super(STDPNet, self).__init__()
        
        beta_dist = torch.distributions.beta.Beta(2,2) # Beta distribution with alpha=2, beta=2. mean=0.5 and variance=1/20. always in [0,1].
        self.layers = list[Layer]()
        for i in range(len(n_features) - 1):
            self.layers.append(
                Layer(
                    Parameter(
                        beta_dist.sample((n_features[i], n_features[i + 1])).requires_grad_(False)
                    ),
                    Parameter(beta_dist.sample((n_features[i + 1],)).requires_grad_(False)),
                    torch.zeros((n_features[i + 1],), requires_grad=False),
                    torch.zeros((n_features[i], n_features[i + 1]), requires_grad=False)
                )
            )

        self.thr = Threshold(1, 0)

    def mark_eligibility(self, pre: BE, post: BE):
        
    
    def forward(self, x: BTE):
        for timestep in range(x.shape[1]): # iterate over timesteps.
            pre_t = x[:, timestep, :]
            for i, layer in enumerate(self.layers):
                d_pot = pre_t @ layer.w + layer.p_t # d_pot: (Batch, n_post)
                post_t = self.thr(d_pot - layer.d_thr) # post_t: (Batch, n_post)
                
                if self.training:
                    post_t = to_synapsewise_epsilon_greedy(post_t) 
                post_t = torch.distributions.binomial.Binomial(1, post_t).sample() # sample an action from the action distribution. (Batch, actions)
        return x


transform = Compose([ToTensor(), Intensity2Latency(num_steps, to_spike=True)])
train_set = MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader[tuple[BTE, int]](
    train_set,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
)

if __name__ == "__main__":
    net = STDPNet().to(device)

    x, y = next(iter(train_loader))
    x = x.to(device)
    y = torch.tensor(y, device=device)
    x = x.view(1, -1)
    y = y.view(1)
    out = net(x)
    print(out)
    print(y)
    print("Done")
