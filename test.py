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
from utils_test import *

test_parser = argparse.ArgumentParser(description='Evaluation of the trained ML-ORAS model')
test_parser.add_argument('--precond', type=bool, default=True, help='Test as a preconditioner')
test_parser.add_argument('--stationary', type=bool, default=True, help='Test as a stationary algorithm')
test_parser.add_argument('--structured', type=bool, default=False, help='Structured or unstructured')
test_parser.add_argument('--PDE', type=str, default='Helmholtz', help='PDE problem')
test_parser.add_argument('--ratio', type=float, default=0.015, help='Lower and upper bound for ratio')
test_parser.add_argument('--TAGConv-k', type=int, default=2, help='TAGConv # of hops')
test_parser.add_argument('--epoch-num', type=int, default=3, help='Epoch number of the network being loaded')
test_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
test_parser.add_argument('--size-unstructured', type=float, default=0.1, help='Lower and upper bound for unstructured size')
test_parser.add_argument('--plot', type=bool, default=True, help='Plot the test grid')
test_parser.add_argument('--model_dir', type=str, default= 'trained_model.pth', help='Directory for loading')
test_parser.add_argument('--size-structured', type=int, default=34, help='Lower and upper bound for structured size')
test_parser.add_argument('--hops', type=int, default=0, help='Learnable hops away from boundary')
test_parser.add_argument('--cut', type=int, default=1, help='RAS delta')
test_args = test_parser.parse_args()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":
    net = 0
    if test_args.structured:
        ratio = test_args.ratio
        n = test_args.size_structured

        old_g = structured(n, n, False)
        grid =  Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops,
                         interior = None , cut=test_args.cut, h = 1/(n+1), nu = 1)
        grid.aggop_gen(ratio = 0.02, cut = 1, node_agg = struct_agg_PWA(n,n,int(n/2),n))
    else:
        lcmin = np.random.uniform(0.042, 0.0421)
        lcmax = np.random.uniform(0.12, 0.121)
        n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
        randomized = True if np.random.rand() < 0.4 else True
        old_g = rand_grid_gen1(randomized = randomized, n = n, lcmin = lcmin, lcmax = lcmax)
        print(old_g.num_nodes)

        grid = Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops,
                          interior = None , cut=test_args.cut, h = 1, nu = 0)

    if test_args.plot:
        grid.plot_agg(size = 0.0, labeling = False, w = 0.1, shade=0.0008)
        plt.show()

    list_test = ['ML_ORAS_res_8_conv_4', 'RAS']#['RAS', 'with Frobenius norm', 'ML_ORAS_res_0_conv_1', 'ML_ORAS_res_0_conv_2', 'ML_ORAS_res_0_conv_4', 'ML_ORAS_res_0_conv_8', 'ML_ORAS_res_0_conv_16']
    directory = test_args.model_dir
    model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=8, num_convs=4, lr = 0.0001)

    if test_args.precond:
        n = grid.aggop[0].shape[0]

        x0 = np.random.random(grid.A.shape[0])
        x0 = x0/((grid.A@x0)**2).sum()**0.5

        b = np.zeros(grid.A.shape[0])

        dict_loss = {}
        dict_precs = {}

        for name in list_test:
            dict_loss[name] = []

            if name == 'Jacobi':
                dict_precs[name] = np.diag(grid.A.diagonal()**(-1))
            elif name[:7] == 'ML_ORAS':
                num_res = int(name[12])

                if num_res == 0:
                    model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=0, num_convs=int(name[19:]), lr = 0.0001)
                else:
                    model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=num_res, num_convs=int(name[19:]), lr = 0.0001)

                model.load_state_dict(torch.load(directory))
                dict_precs[name] = preconditioner(grid, model, precond_type = 'ML_ORAS', u = torch.tensor(x0).unsqueeze(1)).to_dense().numpy()
            elif name == 'with Frobenius norm':
                 model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=8, num_convs=4, lr = 0.0001)

                 model.load_state_dict(torch.load(directory))
                 dict_precs[name] = preconditioner(grid, model, precond_type = 'ML_ORAS', u = torch.tensor(x0).unsqueeze(1)).to_dense().numpy()
            else:
                dict_precs[name] = preconditioner(grid, model, precond_type=name, u = torch.tensor(x0).unsqueeze(1)).to_dense().numpy()

            pyamg.krylov.fgmres(grid.A, b, x0=x0, tol=1e-12,
                       restrt=None, maxiter=int(0.9*n),
                       M=dict_precs[name], callback=None, residuals=dict_loss[name])

        for name in list_test:
            plt.plot(dict_loss[name][:-2], label = name, marker='.')

        plt.xlabel("fGMRES Iteration")
        plt.ylabel("Residual norm")
        plt.yscale('log')
        plt.legend()
        plt.title('GMRES convergence for '+str(int(grid.A.shape[0]))+'-node unstructured grid')
        plt.show()

    if test_args.stationary:
        u = torch.rand(grid.x.shape[0],100).double()
        u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
        K = 10

        dict_enorm = {}
        for name in list_test:
            dict_enorm[name] = []

            if name == 'Jacobi':
                MJ = torch.tensor(np.diag(grid.A.diagonal()**(-1)))
                dict_enorm[name] = test_stationary(grid, model, precond_type = None, u = u, K = K, M = MJ)
            elif name[:7] == 'ML_ORAS':
                num_res = int(name[12])
                if num_res == 0:
                    model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=0, num_convs=int(name[19:]), lr = 0.0001)
                else:
                    model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=num_res, num_convs=int(name[19:]), lr = 0.0001)

                model.load_state_dict(torch.load(directory))
                dict_enorm[name] = test_stationary(grid, model, precond_type = 'ML_ORAS', u = u, K = K)
            elif name == 'with Frobenius norm':
                model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=8, num_convs=4, lr = 0.0001)
                model.load_state_dict(torch.load(directory))
                dict_enorm[name] = test_stationary(grid, model, precond_type = 'ML_ORAS', u = u, K = K)
            else:
                dict_enorm[name] = test_stationary(grid, model, precond_type = name, u = u, K = K)

        for name in list_test:
            plt.plot(dict_enorm[name], label = name, marker='.')

        plt.xlabel("Iteration")
        plt.ylabel("error norm")
        plt.yscale('log')
        plt.title(f'Stationary algorithm: Error norm for {grid.A.shape[0]:.0f}-node unstructured grid')
        plt.legend()
        plt.show()
