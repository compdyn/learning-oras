import networkx as nx
import torch
import torch_geometric as tg
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import shapely
import shapely.geometry as sg
from shapely.ops import cascaded_union
from networkx.drawing.nx_pylab import draw_networkx
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
from torch_geometric.data import Data
import torch_geometric
import copy
import fem
from Unstructured import rand_grid_gen
from torch_geometric.utils import from_scipy_sparse_matrix, from_networkx
import pyamg
from pyamg.aggregation import lloyd_aggregation
import scipy
import time


mpl.rcParams['figure.dpi'] = 300


def structured(n_row, n_col, Neumann = False):
    num_nodes = int(n_row*n_col)

    X = np.array([[i*0.04 for i in range(n_col)] for j in range(n_row)]).flatten()
    Y = np.array([[j*0.04 for i in range(n_col)] for j in range(n_row)]).flatten()
    E = []
    V = np.concatenate((np.expand_dims(X, 1),np.expand_dims(Y, 1)), axis = 1)
    nv = num_nodes
    N = [i for i in range(num_nodes)]

    epsilon = 1
    theta = 1 #param of A matrix

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
    mesh.E = np.zeros((1,4))
    mesh.V = V
    mesh.nv = nv
    mesh.ne = []
    mesh.N = N
    mesh.Edges = Edges
    mesh.num_edges = num_edges


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


    return Old_Grid(AA, mesh)


class Old_Grid(object):
    def __init__(self, A, mesh):
        self.A = A.tocsr()
        self.num_nodes = mesh.nv
        self.mesh = mesh

        active = np.ones(self.num_nodes)
        self.active = active

        self.G = nx.from_scipy_sparse_matrix(self.A, edge_attribute='weight', parallel_edges=False)
        self.x = torch.cat((torch.from_numpy(self.active).unsqueeze(1), \
                        torch.from_numpy(self.active).unsqueeze(1)),dim=1).float()


        edge_index, edge_attr = from_scipy_sparse_matrix(abs(self.A))
        edge_index4P, edge_attr4P = from_scipy_sparse_matrix(self.A)

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
        sub_data = from_networkx(self.G.subgraph(node_list))
        sub_data = Data(x=sub_x, edge_index=sub_data.edge_index, edge_attr= abs(sub_data.weight.float()))

        return sub_data


    def node_hop_neigh(self, node, K):
        return list(nx.single_source_shortest_path(self.G, node, cutoff=K).keys())

    def aggop_gen(self, ratio):
        elem_adj = np.zeros((len(self.mesh.E.tolist()), len(self.mesh.E.tolist())))

        for i, e1 in enumerate(self.mesh.E.tolist()):
            for j, e2 in enumerate(self.mesh.E.tolist()):
                if i!= j:
                    if len(set(e1) - set(e2)) == 1:
                        elem_adj[i,j] = 1

        aggop, seeds = lloyd_aggregation(scipy.sparse.csr_matrix(elem_adj), ratio, maxiter=10_000)
        elem_agg = (aggop, seeds, aggop.indices)
        node_agg = np.zeros((self.mesh.V.shape[0], elem_agg[1].shape[0]))

        for i, e in enumerate(self.mesh.E.tolist()):
            for node in e:
                node_agg[node, elem_agg[-1][i]] = 1

        elem_dict = []
        for e in self.mesh.E.tolist():
            elem_dict.append(set(e))


        self.aggop = (scipy.sparse.csr_matrix(node_agg), 0, elem_dict, elem_agg[-1], 0)
        all_eye = np.eye(self.aggop[0].shape[0])

        self.R = {}
        for i in range(self.aggop[0].shape[1]):
            self.R[i] = all_eye[self.aggop[0].transpose()[i].nonzero()[-1].tolist(), :]

        list_w = []
        for i in range(self.aggop[0].shape[0]):
            w = self.aggop[0][i].indices.shape[0]
            if w>1:
                list_w.append(1/(w-1))
            else:
                list_w.append(1)
        vec_w = np.array(list_w)

        weighted_eye = all_eye * vec_w

        self.R_tilde = {}
        for i in range(self.aggop[0].shape[1]):
            self.R_tilde[i] = weighted_eye[self.aggop[0].transpose()[i].nonzero()[-1].tolist(), :]

        return


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


def structured_2d_poisson_dirichlet(n_pts_x, n_pts_y,
                                        xdim=(0,1), ydim=(0,1),
                                        epsilon=1.0, theta=0.0):
        '''
        Creates a 2D poisson system on a structured grid, discretized using finite elements.
        Dirichlet boundary conditions are assumed.
        Parameters
        ----------
        n_pts_x : integer
          Number of inner points in the x dimension (not including boundary points)
        n_pts_y : integer
          Number of inner points in the y dimension (not including boundary points)
        xdim : tuple (float, float)
          Bounds for domain in x dimension.  Represents smallest and largest x values.
        ydim : tuple (float, float)
          Bounds for domain in y dimension.  Represents smallest and largest y values.
        Returns
        -------
        Grid object with given parameters.
        '''

        x_pts = np.linspace(xdim[0], xdim[1], n_pts_x+2)[1:-1]
        y_pts = np.linspace(xdim[0], ydim[1], n_pts_y+2)[1:-1]
        delta_x = abs(x_pts[1] - x_pts[0])
        delta_y = abs(y_pts[1] - y_pts[0])

        xx, yy = np.meshgrid(x_pts, y_pts)
        xx = xx.flatten()
        yy = yy.flatten()

        grid_x = np.column_stack((xx, yy))
        n = n_pts_x * n_pts_y
        A = sp.lil_matrix((n, n), dtype=np.float64)

        stencil = pyamg.gallery.diffusion_stencil_2d(epsilon=epsilon, theta=theta, type='FD')

        for i in range(n_pts_x):
            for j in range(n_pts_y):
                idx = i + j*n_pts_x

                A[idx, idx] = stencil[1,1]
                has_left = (i>0)
                has_right = (i<n_pts_x-1)
                has_down = (j>0)
                has_up = (j<n_pts_y-1)

                # NSEW connections
                if has_up:
                    A[idx, idx + n_pts_x] = stencil[0, 1]
                if has_down:
                    A[idx, idx - n_pts_x] = stencil[2, 1]
                if has_left:
                    A[idx, idx - 1] = stencil[1, 0]
                if has_right:
                    A[idx, idx + 1] = stencil[1, 2]

                # diagonal connections
                if has_up and has_left:
                    A[idx, idx + n_pts_x - 1] = stencil[0, 0]
                if has_up and has_right:
                    A[idx, idx + n_pts_x + 1] = stencil[0, 2]
                if has_down and has_left:
                    A[idx, idx - n_pts_x - 1] = stencil[2, 0]
                if has_down and has_right:
                    A[idx, idx - n_pts_x + 1] = stencil[2, 2]
        A = A.tocsr()

        return A


def uns_grid(meshsz):

    old_g  = rand_grid_gen(meshsz, 'Helmholtz')

    return old_g


class Grid_PWA():
    def __init__(self, A, mesh, ratio, hops = 0, interior = None, cut=1, h = 1, nu = 1, BC = 'Dirichlet'):
        '''
        Initializes the grid object
        Parameters
        ----------
        A_csr : scipy.sparse.csr_matrix
          CSR matrix representing the underlying PDE
        x : numpy.ndarray
          Positions of the points of each node.  Should have shape (n_pts, n_dim).
        '''

        self.A = A
        self.BC = BC

        self.mesh = mesh
        self.x = self.mesh.V

        if BC == 'Dirichlet':
            self.apply_bc(1e-8)

        self.hops = hops
        self.cut = cut
        self.ratio = ratio
        self.dict_nodes_neighbors_cut = {}
        self.h = h
        self.nu = nu
        self.interior = interior

        modif = scipy.sparse.diags([self.nu * (self.h ** 2) for _ in range(self.A.shape[0])])
        self.A = (1/(self.h ** 2)) * (self.A + modif)


        A_cut = scipy.sparse.csr_matrix(scipy.sparse.identity(self.A.shape[0]))
        for _ in range(self.cut):

            A_cut = A_cut @ self.A

        for n in range(self.A.shape[0]):
            self.dict_nodes_neighbors_cut[n] = set(A_cut[n].nonzero()[-1].tolist())


        self.dict_nodes_neighbors_hop = {}

        A_hop = scipy.sparse.csr_matrix(scipy.sparse.identity(self.A.shape[0]))
        for _ in range(self.hops):

            A_hop = A_hop @ self.A

        for n in range(self.A.shape[0]):
            self.dict_nodes_neighbors_hop[n] = set(A_hop[n].nonzero()[-1].tolist())

        if self.interior is not None:
            self.dict_nodes_neighbors_interior = {}

            A_interior = scipy.sparse.csr_matrix(scipy.sparse.identity(self.A.shape[0]))
            for _ in range(self.interior):

                A_interior = A_interior @ self.A

            for n in range(self.A.shape[0]):
                self.dict_nodes_neighbors_interior[n] = set(A_interior[n].nonzero()[-1].tolist())
        else:
            self.dict_nodes_neighbors_interior = None

        self.aggop_gen(self.ratio, self.cut)


    @property
    def networkx(self):
        return nx.from_scipy_sparse_matrix(self.A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)

    def plot(self, boarders = None, labeling = True, size = 100, w = 1, fsize = 10, ax=None):
        '''
        Plot the nodes and edges of the sparse matrix.
        Parameters
        ----------
        ax : axis
          matplotlib axis
        '''

        graph = self.networkx
        graph.remove_edges_from(nx.selfloop_edges(graph))

        colors = ['r' for _ in range(len(graph.nodes))]

        if boarders is not None:
            for n in boarders:
                colors[n] = 'w'

        if self.x is None:
            positions = None
        else:
            positions = {}
            for node in graph.nodes:
                positions[node] = self.x[node]

        nx.drawing.nx_pylab.draw_networkx(graph, ax=ax, pos=positions, arrows=False,
                                          with_labels=labeling, node_size=size,
                                          width = w, font_size = fsize, node_color = colors)

        plt.axis('equal')

    def plot_agg(self, boarders = True, labeling = True, size = 100, w = 1, fsize = 10,
                 ax=None, color=None, edgecolor='0.5', lw=3, shade = 0.03):
        '''
        Aggregate visualization borrowed/stolen from PyAMG
        (https://github.com/pyamg/pyamg/blob/main/Docs/logo/pyamg_logo.py)
        Parameters
        ----------
        AggOp : CSR sparse matrix
          n x nagg encoding of the aggregates AggOp[i,j] == 1 means node i is in aggregate j
        ax : axis
          matplotlib axis
        color : string
          color of the aggregates
        edgecolor : string
          color of the aggregate edges
        lw : float
          line width of the aggregate edges
        '''

        if boarders:
            boarders = self.boarder_hops()[0]
            self.plot(boarders, labeling, size, w, fsize)

        else:
            self.plot(None, labeling, size, w, fsize)

        if ax is None:
            ax = plt.gca()

        for agg in range(self.aggop[0].shape[1]):
            todraw = []
            for k in self.edge_subs[agg]:
                coords = list(zip(self.x[[k[0], k[1]], 0], self.x[[k[0], k[1]],1]))
                newobj = sg.LineString(coords)
                todraw.append(newobj)

            if self.mesh.E.shape[-1] == 3:
                for e in self.elem_subs[agg]:
                    e = list(e)
                    i = e[0]
                    j1 = e[1]
                    j2 = e[2]

                    coords = list(zip(self.x[[i,j1,j2], 0], self.x[[i,j1,j2],1]))
                    todraw.append(sg.Polygon(coords))

            if self.mesh.E.shape[-1] == 4:
                for k in self.edge_subs[agg]:
                    coords = list(zip(self.x[[k[0], k[1]], 0], self.x[[k[0], k[1]],1]))
                    newobj = sg.LineString(coords)
                    todraw.append(newobj)

                for e in self.elem_subs[agg]:
                    e = list(e)
                    i = e[0]
                    j1 = e[1]
                    j2 = e[2]
                    j3 = e[3]

                    coords = list(zip(self.x[[i,j1,j2], 0], self.x[[i,j1,j2],1]))
                    todraw.append(sg.Polygon(coords))
                    coords = list(zip(self.x[[j3,j1,j2], 0], self.x[[j3,j1,j2],1]))
                    todraw.append(sg.Polygon(coords))
                    coords = list(zip(self.x[[j3,j1,i], 0], self.x[[j3,j1,i],1]))
                    todraw.append(sg.Polygon(coords))
                    coords = list(zip(self.x[[j3,j2,i], 0], self.x[[j3,j2,i],1]))
                    todraw.append(sg.Polygon(coords))

            todraw = shapely.ops.unary_union(todraw)             # union all objects in the aggregate
            todraw = todraw.buffer(shade)                        # expand to smooth
            todraw = todraw.buffer(-shade/2)                     # then contract

            try:
                xs, ys = todraw.exterior.xy                      # get all of the exterior points
                ax.fill(xs, ys, clip_on=False, alpha=0.7)        # fill with a color
            except:
                pass                                             # don't plot singletons
        plt.axis('equal')

    def apply_bc(self, zer):
        if self.mesh.E.shape[-1] == 3:
            max_b = self.A[0].nonzero()[-1][2]
            boundary = [i for i in range(1+max_b)]

        if self.mesh.E.shape[-1] == 4:
            boundary = []
            n_col = int(self.mesh.X.max()/0.04 + 1)
            n_row = int(len(self.mesh.X)/n_col)

            boundary.extend([i for i in range(n_col)])
            boundary.extend([i*n_col for i in range(n_row)])
            boundary.extend([(i+1)*n_col-1 for i in range(n_row)])
            boundary.extend([n_col*n_row - 1 - i for i in range(n_col)])

        self.boundary = boundary

        for n in boundary:
            nzs = self.A[n].nonzero()[-1].tolist()
            for m in nzs:
                self.A[n,m] = zer
                self.A[m,n] = zer
            self.A[n,n] = 1.0

    def learnable_subnodes(self):
        learn_nodes = {}

        for i in range(self.aggop[0].shape[-1]):
            learn_nodes[i] = []
            nz = np.nonzero(self.aggop[0].transpose()[i])[-1].tolist()

            for n in nz:
                cut_neigh_n = list(self.dict_nodes_neighbors_cut[n])
                for m in cut_neigh_n:
                    if self.aggop[0][m, i] == 0:
                        learn_nodes[i].append(m)

                if self.interior is not None:
                    if n in self.list_interior:
                        interior_neigh_n = list(self.dict_nodes_neighbors_interior[n])
                        for m in interior_neigh_n:
                            if self.aggop[0][m, i] == 1 and self.gdata.x[m].item() == 1:
                                learn_nodes[i].append(m)

            learn_nodes[i] = list(set(learn_nodes[i]))

        self.learn_nodes = learn_nodes


    def boarder_hops(self):
        list_boarder = []
        boarder_edges = []
        for e in self.networkx.edges:
            if self.aggop[-1][e[0]] != self.aggop[-1][e[1]]:
                boarder_edges.append(e)
                list_boarder.append(e[0])
                list_boarder.append(e[1])

        list_boarder = list(set(list_boarder))
        list_hops = []

        for n in list_boarder:
            list_hops.extend(list(self.dict_nodes_neighbors_hop[n]))

        list_interior = []
        if self.interior is not None:
            for n in list_boarder:
                list_interior.extend(list(self.dict_nodes_neighbors_interior[n]))

        self.list_interior = list_interior
        return list(set(list_hops)), boarder_edges

    def mask(self):
        sz = self.aggop[0].shape[0]
        if self.boarder_hops()[0] == []:
            return scipy.sparse.csr_matrix(np.eye(sz)*0)
        else:
            list_hops = self.boarder_hops()[0]
            subg = self.networkx.subgraph(list_hops)

            row = np.array(subg.edges)[:,0]
            col = np.array(subg.edges)[:,1]
            data = np.ones(row.shape[0])

            mask_mat = scipy.sparse.csr_matrix((data, (row, col)), shape=(sz, sz))

            return mask_mat

    def data(self):
        sz = self.aggop[0].shape[0]
        masks = self.gmask
        boarder_nodes = self.boarder_hops()[0]
        x = torch.zeros(sz)
        x[boarder_nodes] = 1.0
        edge_index, e_w0 = from_scipy_sparse_matrix(self.A)
        e_w1 = torch.tensor([masks[edge_index[0, i], edge_index[1, i]] for i in range(edge_index[0].shape[0])])
        edge_attr = torch.cat((e_w0.unsqueeze(1), e_w1.unsqueeze(1)), dim = 1)
        data = Data(x = x.unsqueeze(1).float(), edge_index = edge_index, edge_attr = edge_attr.float())

        return data


    def subgrid(self, node_list, idx):
        dict_nodes = {}

        for counter, node in enumerate(node_list):
            dict_nodes[node] = counter

        N = len(node_list)
        mesh2 = copy.deepcopy(self.mesh)

        mesh2.Y = self.mesh.Y[node_list]
        mesh2.X = self.mesh.X[node_list]
        mesh2.V = self.mesh.V[node_list]

        E_list = []

        for ee in self.elem_subs[idx]:
            if len(ee) == 3:
                E_list.append([dict_nodes[ee[0]], dict_nodes[ee[1]], dict_nodes[ee[2]]])
            if len(ee) == 4:
                E_list.append([dict_nodes[ee[0]], dict_nodes[ee[1]], dict_nodes[ee[2]], dict_nodes[ee[3]]])

        mesh2.E = np.array(E_list)
        mesh2.Edge = self.edge_subs[idx]
        mesh2.nv = mesh2.X.shape[0]
        mesh2.num_edges = len(mesh2.Edge)

        mesh2.ne  = mesh2.E.shape[0]
        mesh2.N = [i for i in range(N)]

        A_i, _ = fem.gradgradform(mesh2, kappa=None, f=None, degree=1)
        return A_i


    def set_R_hop(self, cut):
        all_sets = {}
        for i in range(self.aggop[0].shape[1]):
            all_sets[i]= set([])
            nz = self.nz_list[i]
            for n in nz:
                all_sets[i] = all_sets[i].union(self.dict_nodes_neighbors_cut[n])
            all_sets[i] = list(all_sets[i])

        all_eye = np.eye(self.aggop[0].shape[0])

        self.R_hop = {}
        for i in range(self.aggop[0].shape[1]):
            self.R_hop[i] = scipy.sparse.csr_matrix(all_eye[all_sets[i], :])

    def aggop_gen(self, ratio, cut, node_agg=None):
        if self.mesh.E.shape[-1] == 4:
            elem_edges = []
            n_row_elem = int(self.mesh.Y.max()/0.04)
            n_col_elem = int(self.mesh.X.max()/0.04)
            elemg = structured(n_row_elem, n_col_elem)

            elem_dict = []

            for i in range(elemg.mesh.nv):
                e = [i + i//n_col_elem, i + 1 + i//n_col_elem, n_col_elem + i + 2 + i//n_col_elem,
                                                           n_col_elem + i + 1 + i//n_col_elem]
                elem_dict.append(set(e))
        else:
            elem_dict = []
            for e in self.mesh.E.tolist():
                elem_dict.append(set(e))

        self.elem_dict = elem_dict
        if node_agg is None:
            aggop, seeds = lloyd_aggregation(self.A, ratio, maxiter=10_000)
            self.aggop = (aggop, seeds, aggop.indices)
        else:
            self.aggop = node_agg

        self.gmask = self.mask()
        elem_subs = {}
        for _ in range(self.aggop[0].shape[-1]):
            elem_subs[_] = []

        for elem in elem_dict:
            n = self.aggop[-1][list(elem)[0]]
            m = self.aggop[-1][list(elem)[1]]
            k = self.aggop[-1][list(elem)[2]]
            if self.mesh.E.shape[-1] == 4:
                l = self.aggop[-1][list(elem)[3]]
            else:
                l = m

            if n == m and m == k and k == l:
                elem_subs[n].append(list(elem))

        self.elem_subs = elem_subs

        edge_subs = {}
        for _ in range(self.aggop[0].shape[-1]):
                edge_subs[_] = []

        for e in self.networkx.edges:
            if e[0] < e[1]:
                idx0 = np.nonzero(self.aggop[0][e[0]])[-1].item()
                idx1 = np.nonzero(self.aggop[0][e[1]])[-1].item()

                if idx1 == idx0:
                    edge_subs[idx0].append((e[0], e[1]))

        self.edge_subs = edge_subs

        nonzero_list = []
        for i in range(self.aggop[0].shape[1]):
            nonzero_list.append(np.nonzero(self.aggop[0].transpose()[i])[-1].tolist())

        self.nz_list = nonzero_list

        all_eye = np.eye(self.aggop[0].shape[0])

        self.R = {}
        for i in range(self.aggop[0].shape[1]):
            self.R[i] = scipy.sparse.csr_matrix(all_eye[self.nz_list[i], :])

        list_w = []
        for i in range(self.aggop[0].shape[0]):
            w = self.aggop[0][i].indices.shape[0]
            if w>1:
                list_w.append(1/(w-1))
            else:
                list_w.append(1)
        vec_w = np.array(list_w)

        weighted_eye = all_eye * vec_w

        self.R_tilde = {}
        for i in range(self.aggop[0].shape[1]):
            self.R_tilde[i] = scipy.sparse.csr_matrix(weighted_eye[self.nz_list[i], :])

        self.set_R_hop(cut)

        if self.mesh.E.shape[-1] == 2:
            self.A_i = {}
            self.L_i = {}
            L = self.gmask

            for i in range(self.aggop[0].shape[-1]):
                nz = self.nz_list[i]
                num = len(nz)
                a_i = scipy.sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(num, num), format='csr')
                a_i[0,0] = 1
                a_i[-1,-1] = 1
                self.L_i[i] = L[np.ix_(nz, nz)]

        if self.mesh.E.shape[-1] == 3 or self.mesh.E.shape[-1] == 4:
            self.L_i = {}
            L = self.gmask

            for i in range(self.aggop[0].shape[-1]):
                nz = self.nz_list[i]
                self.L_i[i] = L[np.ix_(nz, nz)]

        self.gdata = self.data()
        self.learnable_subnodes()
        if self.BC == 'Dirichlet':
            self.apply_bc(1e-16)
