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
import warnings
import shutil


data_parser = argparse.ArgumentParser(description='Settings for generating data')
data_parser.add_argument('--directory', type=str, default="training_grids", help='Directory to save generated data into')
data_parser.add_argument('--num-data', type=int, default=1000, help='Number of generated problems')
data_parser.add_argument('--structured', type=bool, default=False, help='Structured or unstructured')
data_parser.add_argument('--ratio', type=tuple, default=(0.012, 0.032), help='Lower and upper bound for ratio')
data_parser.add_argument('--size-unstructured', type=tuple, default=(0.2, 0.5), help='Lower and upper bound for unstructured size')
data_parser.add_argument('--size-structured', type=tuple, default=(10, 28), help='Lower and upper bound for structured size')
data_parser.add_argument('--hops', type=int, default=0, help='Learnable hops away from boundary')
data_parser.add_argument('--cut', type=int, default=1, help='RAS delta')
data_args = data_parser.parse_args()


def generate_data(data_args, show_fig = False):
    path = os.path.join(data_args.directory)
    if os.path.exists(path):
        warnings.warn(f'Directory already exists: {path}, deleting...')
        shutil.rmtree(path)

    os.mkdir(path)
    if data_args.structured:
        for i in range(data_args.num_data):
            num_dom = 0

            while num_dom < 2:
                ratio = np.random.uniform(low = data_args.ratio[0], high = data_args.ratio[1])
                n = 10+2*i

                old_g = structured(n, n, False)
                grid =  Grid_PWA(old_g.A, old_g.mesh, ratio = np.random.uniform(low = 0.012, high = 0.032), hops = 0,
                                 interior = None , cut=1, h = 1/(n+1), nu = 1)
                grid.aggop_gen(ratio = 0.1, cut = 1, node_agg = struct_agg_PWA(n,n,int(n/2),n))
                num_dom = grid.aggop[0].shape[-1]

            print(f'Grid {i}, DoF: {grid.A.shape[0]}')

            if show_fig:
                grid.plot_agg(size = 10, fsize = 3)
                plt.show()

            torch.save(grid, data_args.directory+"/grid"+str(i)+".pth")

    else:
        for i in range(data_args.num_data):
            num_node = 0
            num_dom = 0

            while num_dom<2:
                while num_node > 850 or num_node< 85:
                    size = np.random.uniform(low = data_args.size_unstructured[0], high = data_args.size_unstructured[1])
                    ratio = np.random.uniform(low = data_args.ratio[0], high = data_args.ratio[1])

                    lcmin = np.random.uniform(0.03, 0.09)
                    lcmax = np.random.uniform(0.09, 0.5)
                    n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
                    randomized = True if np.random.rand() < 0.4 else False
                    old_g = rand_grid_gen1(randomized = randomized, n = n, lcmin = lcmin, lcmax = lcmax)

                    num_node = old_g.num_nodes

                grid =  Grid_PWA(old_g.A, old_g.mesh, ratio = np.random.uniform(low = 0.012, high = 0.032), hops = 0,
                            interior = None , cut=1, h = 1)

                num_dom = grid.aggop[0].shape[-1]
                num_node = 0

            print(f'Grid {i}, DoF: {old_g.num_nodes}, number of domains: {num_dom}')

            if show_fig:
                grid.plot_agg(size = 10, fsize = 3)
                plt.show()

            torch.save(grid, data_args.directory+"/grid"+str(i)+".pth")

generate_data(data_args, show_fig = False)
