import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class transWGINEConv(MessagePassing):
    def __init__(self, emb_dim, mlp = None, aggr = "add"):
        super(transWGINEConv, self).__init__()
        #multi-layer perceptron
        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, edge_weight = None):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        if edge_weight is not None:
            self_loop_weight = torch.ones(x.size(0))
            self_loop_weight = self_loop_weight.to(edge_weight.device).to(edge_weight.dtype)
            edge_weight = torch.cat((edge_weight, self_loop_weight), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, edge_weight=edge_weight)

    def message(self, x_j, edge_attr, edge_weight):
        return (x_j + edge_attr) if edge_weight is None else (x_j + edge_attr) * edge_weight.view(-1, 1)

    def update(self, aggr_out):
        return self.mlp(aggr_out)