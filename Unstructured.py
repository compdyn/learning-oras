import numpy as np
import sys
sys.path.append('utils')
import matplotlib.pyplot as plt
import scipy
import fem
import pygmsh
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import random
import torch as T
import torch_geometric
import copy
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
from pyamg.gallery import poisson
from torch_geometric.data import Data
from pyamg.aggregation import lloyd_aggregation
import matplotlib as mpl
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr, isspmatrix_csc
from pyamg.graph import lloyd_cluster
from matplotlib.pyplot import figure, text
import torch_geometric.data
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.utils as tgu


mpl.rcParams['figure.dpi'] = 300


class MyMesh:
    def __init__(self, mesh):

        self.nv = mesh.points[:,0:2].shape[0]
        self.X = mesh.points[:,0:1].flatten() * ((self.nv/50)**0.5)
        self.Y = mesh.points[:,1:2].flatten() * ((self.nv/50)**0.5)

        self.E = mesh.cells[1].data
        self.V = mesh.points[:,0:2]

        self.ne = len(mesh.cells[1].data)

        e01 = self.E[:,[0,1]]
        e02 = self.E[:,[0,2]]
        e12 = self.E[:,[1,2]]

        e01 = tuple(map(tuple, e01))
        e02 = tuple(map(tuple, e02))
        e12 = tuple(map(tuple, e12))

        e = list(set(e01).union(set(e02)).union(set(e12)))
        self.N = [i for i in range(self.X.shape[0])]
        self.Edges = e
        self.num_edges = len(e)


def structured(n_row, n_col, Theta):
    num_nodes = int(n_row*n_col)

    X = np.array([[i*0.04 for i in range(n_col)] for j in range(n_row)]).flatten()
    Y = np.array([[j*0.04 for i in range(n_col)] for j in range(n_row)]).flatten()
    E = []
    V = []
    nv = num_nodes
    N = [i for i in range(num_nodes)]

    # Parameters for anisotropy
    epsilon = 1
    theta = 1

    sten = diffusion_stencil_2d(epsilon=epsilon,theta=theta,type='FD')
    AA = stencil_grid(sten, (n_row, n_col), dtype=float, format='csr')

    A = AA.toarray()

    nz_row = np.nonzero(A)[0]
    nz_col = np.nonzero(A)[1]
    e = np.concatenate((np.expand_dims(nz_row,axis=1), np.expand_dims(nz_col, axis=1)), axis=1)
    Edges = list(tuple(map(tuple, e)))
    num_edges = len(Edges)
    g = rand_grid_gen(None)

    mesh = copy.deepcopy(g.mesh)

    mesh.X = X
    mesh.Y = Y
    mesh.E = []
    mesh.V = V
    mesh.nv = nv
    mesh.ne = []
    mesh.N = N
    mesh.Edges = Edges
    mesh.num_edges = num_edges

    Neumann = True
    if Neumann:
        boundary_3 = []
        for i in range(n_row):
            if i == 0 or i == n_row-1:
                boundary_3.extend([i*n_col + j for j in range(n_col)])
            else:
                boundary_3.extend([i*n_col, i*n_col+n_col-1])
        boundary_2 = [0, n_col-1, (n_row-1)*n_col, n_row*n_col-1]

        for i in boundary_3:
            AA[i,i] = 3.0

        for i in boundary_2:
            AA[i,i] = 2.0

    fine_nodes = [i for i in range(num_nodes)]
    return Grid(AA,fine_nodes,[], mesh, Theta)


class Grid(object):
    def __init__(self, A, mesh):
        self.A = A.tocsr()

        self.num_nodes = mesh.nv
        self.mesh = mesh

        active = np.ones(self.num_nodes)
        self.active = active
        self.G = nx.from_scipy_sparse_matrix(self.A, edge_attribute='weight', parallel_edges=False)
        self.x = T.cat((T.from_numpy(self.active).unsqueeze(1),
                        T.from_numpy(self.active).unsqueeze(1)),dim=1).float()

        edge_index, edge_attr = tgu.from_scipy_sparse_matrix(abs(self.A))
        edge_index4P, edge_attr4P = tgu.from_scipy_sparse_matrix(self.A)

        list_neighbours1 = []
        list_neighbours2 = []
        for node in range(self.num_nodes):
            a =  list(self.G.edges(node,data = True))
            l1 = []
            l2 = []
            for i in range(len(a)):
                l1.append(a[i][1])
                l2.append(abs(np.array(list(a[i][-1].values())))[0])

            list_neighbours1.append(l1)
            list_neighbours2.append(l2)

        self.list_neighbours = [list_neighbours1, list_neighbours2]

        self.data = Data(x=self.x, edge_index=edge_index, edge_attr= edge_attr.float())
        self.data4P = Data(x=self.x, edge_index=edge_index4P, edge_attr= edge_attr4P.float())


    def subgrid(self, node_list):
        sub_x = self.x[node_list]
        sub_data = tgu.from_networkx(self.G.subgraph(node_list))
        sub_data = Data(x=sub_x, edge_index=sub_data.edge_index, edge_attr= abs(sub_data.weight.float()))

        return sub_data


    def node_hop_neigh(self, node, K):
        return list(nx.single_source_shortest_path(self.G, node, cutoff=K).keys())


    def plot(self, size, w, labeling, fsize):

        G = nx.from_scipy_sparse_matrix(self.A)
        G.remove_edges_from(nx.selfloop_edges(G))

        mymsh = self.mesh
        pos_dict = {}
        for i in range(mymsh.nv):
            pos_dict[i] = [mymsh.X[i], mymsh.Y[i]]

        colors = [i for i in range(mymsh.nv)]

        for i in range(self.num_nodes):
            colors[i] = 'r'


        draw_networkx(G, pos=pos_dict, with_labels=labeling, node_size=size, \
                      node_color = colors, node_shape = 'o', width = w, font_size = fsize)

        plt.axis('equal')



def grid_subdata_and_plot(node, cutoff, grid_, ploting = False, labeling = True, size = 300.0, w = 1.0):

    node_list = list(nx.single_source_dijkstra_path_length(grid_.G,
                                            node, cutoff = cutoff, weight=None).keys())
    act_coarse_list = []
    sub_x = grid_.data.x[node_list][:,0]
    sub_data = tgu.from_networkx(grid_.G.subgraph(node_list))
    sub_data = Data(x=sub_x, edge_index=sub_data.edge_index,
                    edge_attr= abs(sub_data.weight.float()))

    G = grid_.G.subgraph(node_list)

    mymsh = grid_.mesh
    node_list = list(G.nodes)
    sub_data.x = grid_.data.x[node_list][:,0]

    if ploting:
        pos_dict = {}
        for i in node_list:
            pos_dict[i] = [mymsh.X[i], mymsh.Y[i]]

        colors = [i for i in node_list]

        for i in range(len(node_list)):
            if node_list[i] in list(set(grid_.fine_nodes) - set(grid_.coarse_nodes)):
                colors[i] = 'b'
            if node_list[i] in grid_.coarse_nodes:
                act_coarse_list.append(node_list[i])
                colors[i] = 'r'

        draw_networkx(G, pos=pos_dict, with_labels=labeling, node_size=size,
                      node_color = colors, node_shape = 'o', width = w, font_size=5)


        plt.axis('equal')

    idx_dict = {}
    for i in range(len(node_list)):
        idx_dict[node_list[i]] = i

    after_coarse_list = np.nonzero(sub_data.x == 0).flatten().tolist()

    spmtrx = tgu.to_scipy_sparse_matrix(sub_data.edge_index, edge_attr=sub_data.edge_attr)
    GG = nx.from_scipy_sparse_matrix(spmtrx, edge_attribute='weight', parallel_edges=False)

    return sub_data, node_list, act_coarse_list, after_coarse_list, idx_dict, GG


def set_edge_from_msh(msh):

    edges = msh.E
    array_of_tuples = map(tuple, edges[:,[1,2]])
    t12 = tuple(array_of_tuples)
    array_of_tuples = map(tuple, edges[:,[0,2]])
    t02 = tuple(array_of_tuples)
    array_of_tuples = map(tuple, edges[:,[0,1]])
    t01 = tuple(array_of_tuples)

    set_edge = set(t01).union(set(t02)).union(set(t12))

    return set_edge


def func1(x,y,p):
    x_f = int(np.floor(p.shape[0]*x))
    y_f = int(np.floor(p.shape[1]*y))

    return p[x_f, y_f]


def rand_Amesh_gen1(randomized, n, lcmin, lcmax, kappa=None, gamma=None, PDE='Helmholtz'):
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


    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(hull.points[hull.vertices.tolist()].tolist(), mesh_size=0.1)

        p = 0.05 + 0.6*np.random.random((1000,1000))
        geom.set_mesh_size_callback(

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
            distmin=0.02,
            distmax=0.1,
        )
        field1 = geom.add_boundary_layer(
            nodes_list=nodes_list,
            lcmin=lcmin,
            lcmax=lcmax,
            distmin=0.02,
            distmax=0.1,
        )
        geom.set_background_mesh([field0, field1], operator="Min")
        mesh = geom.generate_mesh()

    mymsh = MyMesh(mesh)

    A,b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1, gamma=gamma , PDE=PDE)
    return A, mymsh


def rand_Amesh_gen(mesh_size, kappa = None, gamma = None, PDE='Helmholtz'):
    num_Qhull_nodes = random.randint(10,45)
    points = np.random.rand(num_Qhull_nodes, 2)
    hull = ConvexHull(points)

    msh_sz = mesh_size
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(hull.points[hull.vertices.tolist()].tolist(), mesh_size=msh_sz)
        mesh = geom.generate_mesh()

    mymsh = MyMesh(mesh)
    A,b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1, gamma=gamma , PDE=PDE)
    return A, mymsh


def rand_grid_gen1(randomized, n, lcmin, lcmax, kappa = None, gamma = None):
    A, mymsh = rand_Amesh_gen1(randomized, n, lcmin, lcmax, kappa = kappa, gamma = gamma)
    return Grid(A,mymsh)

def rand_grid_gen(mesh_sz, kappa = None, gamma = None, PDE='Helmholtz'):
    A, mymsh = rand_Amesh_gen(mesh_sz, kappa = kappa, gamma = gamma, PDE = PDE)
    return Grid(A,mymsh)
