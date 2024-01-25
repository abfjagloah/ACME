import numpy as np
import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool, global_mean_pool
from wgin_conv import WGINConv


class PriorDiscriminator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = Linear(input_dim, input_dim)
        self.l1 = Linear(input_dim, input_dim)
        self.l2 = Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = Sequential(
            Linear(input_dim, input_dim),
            ReLU(),
            Linear(input_dim, input_dim),
            ReLU(),
            Linear(input_dim, input_dim),
            ReLU()
        )
        self.linear_shortcut = Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    
class pre_gnn(torch.nn.Module):
    def __init__(self, encoder, num_features, out_features, pre_dataset):
        super(pre_gnn, self).__init__()
        in_features = pre_dataset.num_features
        num_classes = pre_dataset.num_classes
        self.encoder = encoder
        self.mlp1 = torch.nn.Linear(in_features, num_features)
        self.mlp2 = torch.nn.Linear(out_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data, pool):
        data.x = self.mlp1(data.x)
        y = self.encoder(data, pool)
        y = self.mlp2(y)
        return y
    
    def get_embedding(self, data, pool):
        with torch.no_grad():
            x, _ = self.encoder.encoder(data.x, data.edge_index,data.edge_weight, data.batch, pool)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))

            bn = torch.nn.BatchNorm1d(dim)
            conv = WGINConv(nn)

            self.convs.append(conv)
            self.bns.append(bn)
            

    def forward(self, x, edge_index, edge_weight, batch, pool):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.bns[i](x)
            xs.append(x)
            # if i == 2:
                # feature_map = x2
        if pool == 'mean':
            xpool = [global_mean_pool(x, batch) for x in xs]
        else:
            xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, data_loader,pool, device):
        ret = []
        y = []
        for data in data_loader:
            with torch.no_grad():
                if isinstance(data, list):
                    data = data[0].to(device)
                else:
                    data = data.to(device)

                x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, edge_weight, batch, pool)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

class simclr(nn.Module):
    def __init__(self, dataset, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.embedding_dim = hidden_dim * num_gc_layers

        self.encoder = Encoder(dataset.num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data, pool, device='cuda:0'):
        x, edge_index,edge_weight, batch = data.x, data.edge_index,data.edge_weight, data.batch
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index,edge_weight, batch, pool)
        y = self.proj_head(y)
        return y