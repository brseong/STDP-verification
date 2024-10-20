import torch, torchvision, os
import numpy as np
from typing import Generator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .SpykeTorch.SpykeTorch import utils

from .networks import Mozafari2018
from .dataclasses import ExpInfo
from .functions import Mozafari_train_rl, Mozafari_test, Mozafari_train_unsupervise
from .types import Tensor2D, MNIST_DoG_Data, MNIST_DoG_Target

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train_Mozafari() -> Generator[tuple[Tensor2D,...], None, None]:
    data_root = "data"
    s1c1 = Mozafari2018.generate_transform()
    MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
    MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
    MNIST_loader = DataLoader(MNIST_train, batch_size=1024, num_workers=1, shuffle=False, pin_memory=True)
    MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), num_workers=128, shuffle=False, pin_memory=True)

    mozafari = Mozafari2018()
    def draw_all_weights(): return (mozafari.draw_weights(0),mozafari.draw_weights(1),mozafari.draw_weights(2))
    if ExpInfo.use_cuda:
        mozafari.cuda()
        
    print(mozafari.conv1.weight.shape)
    print(mozafari.conv2.weight.shape)
    print(mozafari.conv3.weight.shape)
    
    # Training The First Layer
    print("Training the first layer")
    if os.path.isfile("saved_l1.net"):
        mozafari.load_state_dict(torch.load("saved_l1.net"))
    else:
        for epoch in (pbar:=tqdm(range(2))):
            iter = 0
            for data,targets in tqdm(MNIST_loader, leave=False):
                pbar.set_description(f"Epoch {epoch}, Iteration {iter}")
                Mozafari_train_unsupervise(mozafari, data, 1)
                yield draw_all_weights()
                iter+=1
        torch.save(mozafari.state_dict(), "saved_l1.net")
    # Training The Second Layer
    print("Training the second layer")
    if os.path.isfile("saved_l2.net"):
        mozafari.load_state_dict(torch.load("saved_l2.net"))
    else:
        for epoch in (pbar:=tqdm(range(4))):
            iter = 0
            for data,targets in tqdm(MNIST_loader, leave=False):
                pbar.set_description(f"Epoch {epoch}, Iteration {iter}")
                Mozafari_train_unsupervise(mozafari, data, 2)
                yield draw_all_weights()
                iter+=1
        torch.save(mozafari.state_dict(), "saved_l2.net")

    # initial adaptive learning rates
    apr = mozafari.stdp3.learning_rate[0][0].item()
    anr = mozafari.stdp3.learning_rate[0][1].item()
    app = mozafari.anti_stdp3.learning_rate[0][1].item()
    anp = mozafari.anti_stdp3.learning_rate[0][0].item()

    adaptive_min = 0
    adaptive_int = 1
    apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
    anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
    app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
    anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

    # perf
    best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
    best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch

    # Training The Third Layer
    print("Training the third layer")
    for epoch in tqdm(range(680)):
        pbar.set_description(f"Epoch {epoch}, Current Train: {best_train[0]}, Best Train: {best_train[0]}, Current Test: {best_test[0]}, Best Test: {best_test[0]}")
        perf_train = np.array([0.0,0.0,0.0])
        data:MNIST_DoG_Data; targets:MNIST_DoG_Target
        for data, targets in tqdm(MNIST_loader, leave=False):
            perf_train_batch = Mozafari_train_rl(mozafari, data, targets)
            assert data.shape[1:] == (15,6,28,28)
            yield draw_all_weights()
            #update adaptive learning rates
            apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
            anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
            app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
            anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
            mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
            perf_train += perf_train_batch
        perf_train /= len(MNIST_loader)
        if best_train[0] <= perf_train[0]:
            best_train = np.append(perf_train, epoch)
        print("Current Train:", perf_train)
        print("   Best Train:", best_train)

        for data,targets in tqdm(MNIST_testLoader, leave=False):
            perf_test = Mozafari_test(mozafari, data, targets)
            if best_test[0] <= perf_test[0]:
                best_test = np.append(perf_test, epoch)
                torch.save(mozafari.state_dict(), "saved.net")
            print(" Current Test:", perf_test)
            print("    Best Test:", best_test)
