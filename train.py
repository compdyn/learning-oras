import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import os
import os.path
from grids import *
import sys
import torch as T
import copy
import random
from NeuralNet import *
from Unstructured import *
import scipy
from grids import *
import time
mpl.rcParams['figure.dpi'] = 300
from ST_CYR import *
import argparse
from utils import *


train_parser = argparse.ArgumentParser(description='Script to train the MLORAS model')
train_parser.add_argument('--num-data', type=int, default=1000, help='Number of training data')
train_parser.add_argument('--num-epoch', type=int, default=4, help='Number of training epochs')
train_parser.add_argument('--mini-batch-size', type=int, default=25, help='Coarsening ratio for aggregation')
train_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
train_parser.add_argument('--TAGConv-k', type=int, default=2, help='TAGConv # of hops')
train_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
train_parser.add_argument('--data-set', type=str, default='training_grids', help='Directory of the training data')
train_parser.add_argument('--save-dir', type=str, default='trained_models', help='Directory of the saved models')
train_parser.add_argument('--K', type=int, default=4, help='Number of iterations in the loss function')
train_args = train_parser.parse_args()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":
    save_dir = train_args.save_dir
    train_args.save_directory = os.path.join(save_dir)
    if not os.path.exists(train_args.save_directory):
        os.mkdir(train_args.save_directory)

    if not os.path.exists(train_args.data_set):
        raise RuntimeError(f'Training directory does not exist: {train_args.data_set}.')
    list_grids = []
    for i in range(train_args.num_data):
        g = torch.load(train_args.data_set+"/grid"+str(i)+".pth")
        list_grids.append(g)

    num_res = 8
    model = mloras_net (dim = 128, K = 2, num_res = num_res, num_convs = 4, lr = 0.0001)
    loss_list = []

    for itr in range(train_args.num_epoch):
        for _ in range(int(train_args.num_data/train_args.mini_batch_size)):
            indices = np.random.choice(train_args.num_data, size=train_args.mini_batch_size, replace=False)
            loss = 0
            model.optimizer.zero_grad()

            for i in indices:
                grid = list_grids[i]

                u = torch.randn(grid.x.shape[0],500).double()
                u = u/(((u**2).sum(0))**0.5).unsqueeze(0)

                loss += stationary_max(grid, model, u = u, K = train_args.K, precond_type='ML_ORAS')

            loss_list.append(loss.item())
            print(f'Epoch {itr}, loss: {loss.item():.6f}')
            loss.backward()
            model.optimizer.step()
        torch.save(model.state_dict(), train_args.save_directory+"/model_epoch"+str(itr)+".pth")

    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss vs. Iteration')
    plt.show()
