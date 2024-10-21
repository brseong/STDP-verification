import torch
import numpy as np

from .dataclasses import expr_info_global
from .networks import Mozafari2018

def Mozafari_train_unsupervise(network:Mozafari2018, data:torch.Tensor, layer_idx:int):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if expr_info_global.use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)

def Mozafari_train_rl(network:Mozafari2018, data:torch.Tensor, target:torch.Tensor):
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if expr_info_global.use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)

def Mozafari_test(network:Mozafari2018, data:torch.Tensor, target:torch.Tensor):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if expr_info_global.use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)