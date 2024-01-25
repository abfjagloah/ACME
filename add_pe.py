import torch
import torch.utils.data
import copy
from torch_geometric.utils import to_scipy_sparse_matrix,degree
from scipy import sparse as sp
import networkx as nx
import numpy as np

def init_positional_encoding(data, device, pos_enc_dim,type_init):
    # data = data_in
    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = to_scipy_sparse_matrix(data.edge_index,None,data.num_nodes)
        Dinv = sp.diags((degree(data.edge_index[1]).cpu().numpy()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1).to(device)
        
    
    elif type_init == 'lap':
        # Laplacian
        A = to_scipy_sparse_matrix(data.edge_index,data.edge_attr,data.num_nodes)
        N = sp.diags((degree(data.edge_index[1]).cpu().numpy()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(data.num_nodes) - N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        PE = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float().to(device)
        data.x = torch.cat([data.x,PE],dim=1)
        # g.ndata['eigvec'] = g.ndata['pos_enc']

    return PE