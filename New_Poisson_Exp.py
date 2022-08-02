#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:59:20 2022

@author: alitaghibakhshi
"""

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
import scipy.sparse as sparse

train_parser = argparse.ArgumentParser(description='Settings for training machine learning for ORAS')

train_parser.add_argument('--num-data', type=int, default=100, help='Number of training data')
train_parser.add_argument('--num-epoch', type=int, default=10, help='Number of training epochs')
train_parser.add_argument('--mini-batch-size', type=int, default=25, help='Coarsening ratio for aggregation')
train_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
train_parser.add_argument('--TAGConv-k', type=int, default=2, help='TAGConv # of hops')
train_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
train_parser.add_argument('--data-set', type=str, default='Grids_Poisson', help='Directory of the training data')
train_parser.add_argument('--save-dir', type=str, default='Grids_For_Poisson', help='Directory of the saved models')
train_parser.add_argument('--K', type=int, default=4, help='Number of iterations in the loss function')

train_args = train_parser.parse_args()


test_parser = argparse.ArgumentParser(description='Settings for training machine learning for ORAS')

test_parser.add_argument('--precond', type=bool, default=True, help='Test as a preconditioner')
test_parser.add_argument('--stationary', type=bool, default=True, help='Test as a stationary algorithm')
test_parser.add_argument('--structured', type=bool, default=False, help='Structured or unstructured')
test_parser.add_argument('--PDE', type=str, default='Poisson', help='PDE problem')
test_parser.add_argument('--BC', type=str, default='Dirichlet', help='TBoundary conditions')
test_parser.add_argument('--ratio', type=float, default=0.015, help='Lower and upper bound for ratio')
test_parser.add_argument('--TAGConv-k', type=int, default=2, help='TAGConv # of hops')
test_parser.add_argument('--epoch-num', type=int, default=9, help='Epoch number of the network being loaded')
test_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
test_parser.add_argument('--size-unstructured', type=float, default=0.1, help='Lower and upper bound for unstructured size')
test_parser.add_argument('--plot', type=bool, default=True, help='Plot the test grid')
test_parser.add_argument('--model_dir', type=str, default= 'Grids_For_Poisson', help='Directory for loading ')
test_parser.add_argument('--size-structured', type=int, default=34, help='Lower and upper bound for structured size')
test_parser.add_argument('--hops', type=int, default=0, help='Learnable hops away from boundary')
test_parser.add_argument('--cut', type=int, default=1, help='RAS delta')

test_args = test_parser.parse_args()


def Poisson_make(mesh, kappa=None, f=None, degree=1, kap=1.0):
    """Finite element discretization of a Poisson problem.

    - div . kappa(x,y) grad u = f(x,y)

    Parameters
    ----------
    V : ndarray
        nv x 2 list of coordinates

    E : ndarray
        ne x 3 or 6 list of vertices

    kappa : function
        diffusion coefficient, kappa(x,y) with vector input

    fa : function
        right hand side, f(x,y) with vector input

    degree : 1 or 2
        polynomial degree of the bases (assumed to be Lagrange locally)

    Returns
    -------
    A : sparse matrix
        finite element matrix where A_ij = <kappa grad phi_i, grad phi_j>

    b : array
        finite element rhs where b_ij = <f, phi_j>

    Notes
    -----
        - modepy is used to generate the quadrature points
          q = modepy.XiaoGimbutasSimplexQuadrature(4,2)

    Example
    -------
    >>> import numpy as np
    >>> import fem
    >>> import scipy.sparse.linalg as sla
    >>> V = np.array(
        [[  0,  0],
         [  1,  0],
         [2*1,  0],
         [  0,  1],
         [  1,  1],
         [2*1,  1],
         [  0,2*1],
         [  1,2*1],
         [2*1,2*1],
        ])
    >>> E = np.array(
        [[0,1,3],
         [1,2,4],
         [1,4,3],
         [2,5,4],
         [3,4,6],
         [4,5,7],
         [4,7,6],
         [5,8,7]])
    >>> A, b = fem.poissonfem(V, E)
    >>> print(A.toarray())
    >>> print(b)
    >>> f = lambda x, y : 0*x + 1.0
    >>> g = lambda x, y : 0*x + 0.0
    >>> g1 = lambda x, y : 0*x + 1.0
    >>> tol = 1e-12
    >>> X, Y = V[:,0], V[:,1]
    >>> id1 = np.where(abs(Y) < tol)[0]
    >>> id2 = np.where(abs(Y-2) < tol)[0]
    >>> id3 = np.where(abs(X) < tol)[0]
    >>> id4 = np.where(abs(X-2) < tol)[0]
    >>> bc = [{'id': id1, 'g': g},
              {'id': id2, 'g': g},
              {'id': id3, 'g': g1},
              {'id': id4, 'g': g}]
    >>> A, b = fem.poissonfem(V, E, f=f, bc=bc)
    >>> u = sla.spsolve(A, b)
    >>> print(A.toarray())
    >>> print(b)
    >>> print(u)
    """
    if degree not in [1, 2]:
        raise ValueError('degree = 1 or 2 supported')

    if f is None:
        def f(x, y):
            return 0.0

    if kappa is None:
        def kappa(x, y):
            return kap
            # if x>0.5:
            #     return 1.
            # else:
            #     return 1000.


    if not callable(f) or not callable(kappa):
        raise ValueError('f, kappa must be callable functions')

    ne = mesh.ne

    if degree == 1:
        E = mesh.E
        V = mesh.V
        X = mesh.X
        Y = mesh.Y

    if degree == 2:
        E = mesh.E2
        E = E.astype(int)
        V = mesh.V2
        X = mesh.X2
        Y = mesh.Y2

    # allocate sparse matrix arrays
    m = 3 if degree == 1 else 6
    AA = np.zeros((ne, m**2))
    IA = np.zeros((ne, m**2), dtype=int)
    JA = np.zeros((ne, m**2), dtype=int)
    bb = np.zeros((ne, m))
    ib = np.zeros((ne, m), dtype=int)
    jb = np.zeros((ne, m), dtype=int)

    # Assemble A and b
    for ei in range(0, ne):
        # Step 1: set the vertices and indices
        K = E[ei, :]
        x0, y0 = X[K[0]], Y[K[0]]
        x1, y1 = X[K[1]], Y[K[1]]
        x2, y2 = X[K[2]], Y[K[2]]

        # Step 2: compute the Jacobian, inv, and det
        J = np.array([[x1 - x0, x2 - x0],
                      [y1 - y0, y2 - y0]])
        invJ = np.linalg.inv(J.T)
        detJ = np.linalg.det(J)

        if degree == 1:
            # Step 3, define the gradient of the basis
            dbasis = np.array([[-1, 1, 0],
                               [-1, 0, 1]])

            # Step 4
            dphi = invJ.dot(dbasis)

            # Step 5, 1-point gauss quadrature

            Aelem = kappa(X[K].mean(), Y[K].mean()) * (detJ / 2.0) * (dphi.T).dot(dphi)
            

            # Step 6, 1-point gauss quadrature
            belem = f(X[K].mean(), Y[K].mean()) * (detJ / 6.0) * np.ones((3,))

        if degree == 2:
            ww = np.array([0.44676317935602256, 0.44676317935602256, 0.44676317935602256,
                           0.21990348731064327, 0.21990348731064327, 0.21990348731064327])
            xy = np.array([[-0.10810301816807008, -0.78379396366385990],
                           [-0.10810301816806966, -0.10810301816807061],
                           [-0.78379396366386020, -0.10810301816806944],
                           [-0.81684757298045740, -0.81684757298045920],
                           [0.63369514596091700, -0.81684757298045810],
                           [-0.81684757298045870, 0.63369514596091750]])
            xx, yy = (xy[:, 0]+1)/2, (xy[:, 1]+1)/2
            ww *= 0.5

            Aelem = np.zeros((m, m))
            belem = np.zeros((m,))

            for w, x, y in zip(ww, xx, yy):
                # Step 3
                basis = np.array([(1-x-y)*(1-2*x-2*y),
                                  x*(2*x-1),
                                  y*(2*y-1),
                                  4*x*(1-x-y),
                                  4*x*y,
                                  4*y*(1-x-y)])

                dbasis = np.array([
                    [4*x + 4*y - 3, 4*x-1,     0, -8*x - 4*y + 4, 4*y,           -4*y],
                    [4*x + 4*y - 3,     0, 4*y-1,           -4*x, 4*x, -4*x - 8*y + 4]
                ])

                # Step 4
                dphi = invJ.dot(dbasis)

                # Step 5
                xt, yt = J.dot(np.array([x, y])) + np.array([x0, y0])
                                    
                Aelem += (detJ / 2) * w * kappa(xt, yt) * dphi.T.dot(dphi)

                    
                # Step 6
                belem += (detJ / 2) * w * f(xt, yt) * basis

        # Step 7
        AA[ei, :] = Aelem.ravel()
        IA[ei, :] = np.repeat(K[np.arange(m)], m)
        JA[ei, :] = np.tile(K[np.arange(m)], m)
        bb[ei, :] = belem.ravel()
        ib[ei, :] = K[np.arange(m)]
        jb[ei, :] = 0

    # convert matrices
    A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
    A.sum_duplicates()
    b = sparse.coo_matrix((bb.ravel(), (ib.ravel(), jb.ravel()))).toarray().ravel()
    
    
    # A = A.tocsr()
    return A, b

def rand_Amesh_gen1(randomized, n, min_, min_sz, lcmin, lcmax, distmin, distmax, kappa=None, kap=1.0):
    
    num_Qhull_nodes = random.randint(3, 45)
    if randomized:
        points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
        hull = ConvexHull(points)
    else:
        points = []
    
        for i in range(1,n+1):
            points.append([0.5+0.2*np.cos(i*2*np.pi/n + np.pi/n), 0.5+0.2*np.sin(i*2*np.pi/n + np.pi/n)])
        hull = ConvexHull(points)
        points = np.array(points)
    
    msh_sz = 0.1 #0.1*random.random()+0.1
    
        
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            
                hull.points[hull.vertices.tolist()].tolist()
                
            ,
            mesh_size=msh_sz,
        )
        
        p = 0.6 + 0.5*np.random.random((500,500))
        geom.set_mesh_size_callback(
            #lambda dim, tag, x, y, z: func(x, y, points,min_dist, thresh, min_sz)
            lambda dim, tag, x, y, z: func1(x, y, p)
        )
        
        n_edge = len(poly.curves)
        list_edge_idx = np.random.randint(0, n_edge, np.random.randint(1,3,1).item())
        edges_list = [poly.curves [i] for i in list_edge_idx]
        
        n_points = len(poly.points)
        list_point_idx = np.random.randint(0, n_points, np.random.randint(1,5,1).item())
        nodes_list = [poly.points [i] for i in list_point_idx]
        
        field0 = geom.add_boundary_layer(
            edges_list=edges_list,
            lcmin=lcmin,
            lcmax=lcmax,
            distmin=distmin,
            distmax=distmax,
        )
        field1 = geom.add_boundary_layer(
            nodes_list=nodes_list,
            lcmin=lcmin,
            lcmax=lcmax,
            distmin=distmin,
            distmax=distmax,
        )
        geom.set_background_mesh([field0, field1], operator="Min")
  
        mesh = geom.generate_mesh()
        
    mymsh = MyMesh(mesh)
    # points = mymsh.V
    # tri = Delaunay(points)
    # plt.triplot(points[:,0], points[:,1], tri.simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    
    A,b = Poisson_make(mymsh, kappa=None, f=None, degree=1, kap=kap)
    
    return A, mymsh

def rand_grid_gen1(randomized, n, min_, min_sz, lcmin, lcmax,distmin, distmax, kappa = None, kap=1.0):
    
    A, mymsh = rand_Amesh_gen1(randomized, n, min_, min_sz, lcmin, lcmax, distmin, 
                               distmax, kappa = kappa, kap=kap)
    
    rand_grid = Grid(A,mymsh)
    
    return rand_grid



def stat_test(grid,model):
    
    u = torch.rand(grid.x.shape[0],100).double()
    u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
    K = 10
    
    
    dict_enorm = {}
    # list_test = ['RAS', 'ML_ORAS_res_0_conv_4', 'ML_ORAS_res_1_conv_4', 'ML_ORAS_res_2_conv_4', 'ML_ORAS_res_4_conv_4', 'ML_ORAS_res_8_conv_4', 'ML_ORAS_res_8_conv_1', 'ML_ORAS_res_8_conv_2', 'ML_ORAS_res_8_conv_8', 'ML_ORAS_res_8_conv_16']
    list_test = ['RAS',  'ML_ORAS_res_8_conv_4']#['RAS', 'ML_ORAS_res_0_conv_1', 'ML_ORAS_res_0_conv_2', 'ML_ORAS_res_0_conv_4', 'ML_ORAS_res_0_conv_8', 'ML_ORAS_res_0_conv_16']
    #'with Frobenius norm', 'ML_ORAS_nores5', 'ML_ORAS_nores10', 'ML_ORAS_nores15', 'ML_ORAS_nores20']
    for name in list_test:
        
        dict_enorm[name] = []

        if name == 'Jacobi':
            MJ = torch.tensor(np.diag(grid.A.diagonal()**(-1)))
            dict_enorm[name] = test_stationary(grid, model, precond_type = None, u = u, K = K, M = MJ)
        
        elif name[:7] == 'ML_ORAS':

            num_res = int(name[12])
            model = mloras_net (dim = 128, K = 2, num_res = 8, num_convs = 4, lr = 0.0001, res = True)
            # if num_res == 0:
               
            #     model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=0, num_convs=int(name[19:]), lr = 0.0001, res=False)
            # else:
            #     model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=num_res, num_convs=int(name[19:]), lr = 0.0001, res=True)

            # directory  = parent_dir + "Models_for_" + test_args.model_dir + "/model_epoch"+str(test_args.epoch_num)+".pth" 
            directory = '/Users/alitaghibakhshi/PycharmProjects/ML_OSM/mloras_neurips/Models/Models_for_Grids_max_res_8_conv_4/model_epoch9.pth'
            model.load_state_dict(torch.load(directory))
           
            dict_enorm[name] = test_stationary(grid, model, precond_type = 'ML_ORAS', u = u, K = K)                  

        elif name == 'with Frobenius norm':
            model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=8, num_convs=4, lr = 0.0001, res=True)

            # model = SmallNet(test_args.dim, test_args.TAGConv_k, 8, 0.0001)
            directory = 'Models/Models_for_Grids_Frob_loss/model_epoch3.pth'
            model.load_state_dict(torch.load(directory))
            dict_enorm[name] = test_stationary(grid, model, precond_type = 'ML_ORAS', u = u, K = K) 
            
        else:
            dict_enorm[name] = test_stationary(grid, model, precond_type = name, u = u, K = K) 
        
    
        
    return dict_enorm
    
    
def gmres_test(grid, model):
    
    n = grid.aggop[0].shape[0]
    
    x0 = np.random.random(grid.A.shape[0])
    x0 = x0/((grid.A@x0)**2).sum()**0.5
    
    b = np.zeros(grid.A.shape[0])
    
    dict_loss = {}
    dict_precs = {}
    # list_test = ['RAS', 'ML_ORAS_res_0_conv_4', 'ML_ORAS_res_1_conv_4', 'ML_ORAS_res_2_conv_4', 'ML_ORAS_res_4_conv_4', 'ML_ORAS_res_8_conv_4', 'ML_ORAS_res_8_conv_1', 'ML_ORAS_res_8_conv_2', 'ML_ORAS_res_8_conv_8', 'ML_ORAS_res_8_conv_16']
    list_test = ['RAS', 'ML_ORAS_res_8_conv_4']#['RAS', 'ML_ORAS_res_0_conv_1', 'ML_ORAS_res_0_conv_2', 'ML_ORAS_res_0_conv_4', 'ML_ORAS_res_0_conv_8', 'ML_ORAS_res_0_conv_16']

                 #'with Frobenius norm', 'ML_ORAS_nores5', 'ML_ORAS_nores10', 'ML_ORAS_nores15', 'ML_ORAS_nores20']
    
    for name in list_test:
        
        dict_loss[name] = []
        
        if name == 'Jacobi':
            dict_precs[name] = np.diag(grid.A.diagonal()**(-1))
            
        elif name[:7] == 'ML_ORAS':
            
            num_res = int(name[12])
            
            if num_res == 0:
                
                model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=0, num_convs=int(name[19:]), lr = 0.0001, res=False)
            else:
                model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=num_res, num_convs=int(name[19:]), lr = 0.0001, res=True)

            # directory  = parent_dir + "Models_for_" + test_args.model_dir + "/model_epoch"+str(test_args.epoch_num)+".pth" 
            directory = '/Users/alitaghibakhshi/PycharmProjects/ML_OSM/mloras_neurips/Models/Models_for_Grids_max_res_8_conv_4/model_epoch9.pth'

            model.load_state_dict(torch.load(directory))
            dict_precs[name] = preconditioner(grid, model, precond_type = 'ML_ORAS', u = torch.tensor(x0).unsqueeze(1)).to_dense().numpy()
            
        elif name == 'with Frobenius norm':
             model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res=8, num_convs=4, lr = 0.0001, res=True)

             # model = SmallNet(test_args.dim, test_args.TAGConv_k, 8, 0.0001)
             directory = 'Models/Models_for_Grids_Frob_loss/model_epoch3.pth'
             model.load_state_dict(torch.load(directory))
             dict_precs[name] = preconditioner(grid, model, precond_type = 'ML_ORAS', u = torch.tensor(x0).unsqueeze(1)).to_dense().numpy()

        else:
            dict_precs[name] = preconditioner(grid, model, precond_type=name, u = torch.tensor(x0).unsqueeze(1)).to_dense().numpy()
            
        pyamg.krylov.fgmres(grid.A, b, x0=x0, tol=1e-12, 
                   restrt=None, maxiter=int(0.9*n),
                   M=dict_precs[name], callback=None, residuals=dict_loss[name])
    
        
    return dict_loss




model = mloras_net (dim = 128, K = 2, num_res = 8, num_convs = 4, lr = 0.0001, res = True)
parent_dir = "Models/" 
# directory  = parent_dir + "Models_for_" + test_args.model_dir + "/model_epoch"+str(test_args.epoch_num)+".pth" 
directory = '/Users/alitaghibakhshi/PycharmProjects/ML_OSM/mloras_neurips/Models/Models_for_Grids_max_res_8_conv_4/model_epoch9.pth'
model.load_state_dict(torch.load(directory))
model.eval()
dict_diff_kap = {}
kapsize = np.linspace(1.0, 3.0,10)
for i in range(1):
    
    kap = kapsize[i]
    lcmin = np.random.uniform(0.062, 0.0621)
    lcmax = np.random.uniform(0.12, 0.121)
    n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
    randomized = True if np.random.rand() < 0.4 else True
    num_nodes_grid= 0
    while num_nodes_grid < 1400 or num_nodes_grid > 1600:
        old_g = rand_grid_gen1(randomized = randomized, n = n,min_ = 0.6,min_sz = 0.5,
                      lcmin = lcmin, lcmax = lcmax, distmin = 0.02, distmax = 0.1, kap=kap)
        num_nodes_grid = old_g.num_nodes
    print(num_nodes_grid)

    grid =  Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops, 
                      interior = None , cut=test_args.cut, h = 1, nu = 0, BC = test_args.BC)   
    loc_dict = {}
    loc_dict['stationary'] = stat_test(grid, model)
    loc_dict['gmres'] = gmres_test(grid, model)
    dict_diff_kap[kap] = loc_dict
    
# torch.save(dict_diff_kap, 'dict_diff_kap.pth')



#Experiment C2#
# model = mloras_net (dim = 128, K = 2, num_res = 8, num_convs = 4, lr = 0.0001, res = True)
# parent_dir = "Models/" 
# # directory  = parent_dir + "Models_for_" + test_args.model_dir + "/model_epoch"+str(test_args.epoch_num)+".pth" 
# directory = '/Users/alitaghibakhshi/PycharmProjects/ML_OSM/mloras_neurips/Models/Models_for_Grids_max_res_8_conv_4/model_epoch9.pth'
# model.load_state_dict(torch.load(directory))
# model.eval()
# list_size = [2**i for i in range(7,16)]
# dict_discont_size = {}#torch.load('dict_discont_size.pth')

# this_size = list_size[4]

# lcmin = np.random.uniform(0.072, 0.0721)
# lcmax = np.random.uniform(0.10, 0.101)
# n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
# randomized = True if np.random.rand() < 0.4 else True
# kap = 1.#(4.5*np.random.rand(1)+0.5)[0]
# num_nodes_grid= 0
# while num_nodes_grid < this_size*0.8 or num_nodes_grid > this_size*1.2:
#     old_g = rand_grid_gen1(randomized = randomized, n = n,min_ = 0.6,min_sz = 0.5,
#                   lcmin = lcmin, lcmax = lcmax, distmin = 0.02, distmax = 0.1)
#     num_nodes_grid = old_g.num_nodes
#     print(num_nodes_grid)

# grid =  Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops, 
#                   interior = None , cut=test_args.cut, h = 1, nu = 0, BC = test_args.BC)   
# loc_dict = {}
# loc_dict['stationary'] = stat_test(grid, model)
# loc_dict['gmres'] = gmres_test(grid, model)
# dict_discont_size[this_size] = loc_dict
# # torch.save(dict_discont_size, 'dict_discont_size.pth')

dict_enorm = loc_dict['stationary']
dict_loss  = loc_dict['gmres']

list_test = ['RAS', 'ML_ORAS_res_8_conv_4']
label = {'RAS': 'RAS', 'ML_ORAS_res_8_conv_4':'MLORAS'}
plt.figure()
for name in list_test:
    
    plt.plot(dict_enorm[name], label = label[name], marker='.')
    
# torch.save(dict_enorm, '/Users/alitaghibakhshi/PycharmProjects/ML_OSM/ORAS_paper/Data/'+'uns_stationary'+str(grid.aggop[0].shape[0])+'.pth')

plt.xlabel("Iteration")
plt.ylabel("error norm")
plt.yscale('log')
# plt.ylim([1e-3, 1])
plt.title('Stationary algorithm: Error norm for '+str(int(grid.A.shape[0]))+'-node unstructured grid')
plt.legend()
# plt.savefig('/Users/alitaghibakhshi/PycharmProjects/ML_OSM/Tex_paper/Figs/uns_conv'+str(int(grid.A.shape[0]))+'.pdf')
plt.show()

plt.figure()
for name in list_test:
    
    plt.plot(dict_loss[name][:-2], label = label[name], marker='.')

# torch.save(dict_loss, '/Users/alitaghibakhshi/PycharmProjects/ML_OSM/ORAS_paper/Data/'+'uns_gmres'+str(grid.aggop[0].shape[0])+'.pth')

plt.xlabel("fGMRES Iteration")
plt.ylabel("Residual norm")
plt.yscale('log')
plt.legend()
plt.title('GMRES convergence for '+str(int(grid.A.shape[0]))+'-node unstructured grid')
# plt.savefig('/Users/alitaghibakhshi/PycharmProjects/ML_OSM/Tex_paper/Figs/uns_gmres'+str(int(grid.A.shape[0]))+'.pdf')
plt.show()