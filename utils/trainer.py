import torch, torchvision, os
import numpy as np
from typing import Generator
from torch.utils.data import DataLoader

from .SpykeTorch.SpykeTorch import utils

from .networks import Mozafari2018
from .dataclasses import ExpInfo
from .functions import Mozafari_train_rl, Mozafari_test, Mozafari_train_unsupervise

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train_Mozafari() -> Generator:
    data_root = "data"
    s1c1 = Mozafari2018.generate_transform()
    MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
    MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
    MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=False)
    MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

    mozafari = Mozafari2018()
    if ExpInfo.use_cuda:
        mozafari.cuda()

    # Training The First Layer
    print("Training the first layer")
    if os.path.isfile("saved_l1.net"):
        mozafari.load_state_dict(torch.load("saved_l1.net"))
    else:
        for epoch in range(2):
            print("Epoch", epoch)
            iter = 0
            for data,targets in MNIST_loader:
                print("Iteration", iter)
                Mozafari_train_unsupervise(mozafari, data, 1)
                yield mozafari.draw_weights(0)
                print("Done!")
                iter+=1
        torch.save(mozafari.state_dict(), "saved_l1.net")
    # Training The Second Layer
    print("Training the second layer")
    if os.path.isfile("saved_l2.net"):
        mozafari.load_state_dict(torch.load("saved_l2.net"))
    else:
        for epoch in range(4):
            print("Epoch", epoch)
            iter = 0
            for data,targets in MNIST_loader:
                print("Iteration", iter)
                Mozafari_train_unsupervise(mozafari, data, 2)
                yield mozafari.draw_weights(1)
                print("Done!")
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
    for epoch in range(680):
        print("Epoch #:", epoch)
        perf_train = np.array([0.0,0.0,0.0])
        for data,targets in MNIST_loader:
            perf_train_batch = Mozafari_train_rl(mozafari, data, targets)
            yield mozafari.draw_weights(2)
            print(perf_train_batch)
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

        for data,targets in MNIST_testLoader:
            perf_test = Mozafari_test(mozafari, data, targets)
            if best_test[0] <= perf_test[0]:
                best_test = np.append(perf_test, epoch)
                torch.save(mozafari.state_dict(), "saved.net")
            print(" Current Test:", perf_test)
            print("    Best Test:", best_test)
