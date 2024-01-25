import copy
import random
import numpy as np
import os
import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import VGAE, GINConv, global_add_pool
from torch_geometric.utils import  subgraph, k_hop_subgraph
    
class GIN_WeightEncoder(torch.nn.Module):
    def __init__(self, dataset, dim, num_gc_layers):
        super().__init__()

        num_features = dataset.num_features
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            else:
                mlp = nn.Sequential(nn.Linear(num_features, dim), nn.ReLU(), nn.Linear(dim, dim))

            bn = torch.nn.BatchNorm1d(dim)
            conv = GINConv(mlp)

            self.convs.append(conv)
            self.bns.append(bn)

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

    def forward(self, data):        
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
        
        # x_mean = self.encoder_mean(x)
        # x_std = self.encoder_std(x)
        # gaussian_noise = torch.randn(x_mean.shape).to(x.device)
        # x_final = gaussian_noise * x_std + x_mean
        return x
    
class View_expert(torch.nn.Module):
    def __init__(self,  dataset, hidden_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size
        edge_features = (2*hidden_dim + dataset.num_edge_features)
        self.node_expert = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.mask_expert1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.mask_expert2 = nn.Sequential(nn.Linear(batch_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.edge_expert = nn.Sequential(nn.Linear(edge_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.subgraph_expert = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
    
    def forward(self, x_, data):
        p1 = self.node_expert(x_)

        p2_1 = self.mask_expert1(x_)
        token = global_add_pool(data.x, data.batch).T
        if token.shape[1] != self.batch_size:
            pad_num = self.batch_size - token.shape[1]
            token = torch.nn.functional.pad(token,(0, pad_num), mode='constant', value=0)
        p2_2 = self.mask_expert2(token)

        node_attr = x_[data.edge_index]
        edge_embed = torch.cat([node_attr[0],node_attr[1]],1)
        if data.edge_attr != None:
            edge_embed = torch.cat([edge_embed,data.edge_attr],1)
        p3 = self.edge_expert(edge_embed)

        p4 = self.subgraph_expert(x_)

        return p1, p2_1, p2_2, p3, p4
    
class CompositionalViewGenerator(VGAE):
        def __init__(self, dataset, hidden_dim, num_gc_layers, batch_size):
            self.batch_size = batch_size
            encoder = GIN_WeightEncoder(dataset, hidden_dim, num_gc_layers)
            decoder = View_expert(dataset, hidden_dim, batch_size)
            super().__init__(encoder=encoder, decoder=decoder)
            
                    
        def forward(self, data_in, requires_grad, hop):
            data = copy.deepcopy(data_in)
            
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
            edge_attr = None
            if data.edge_attr is not None:
                edge_attr = data.edge_attr

            data.x = data.x.float()
            x = x.float()
            x.requires_grad = requires_grad

            x_ = self.encoder(data)
            p1, p2_1, p2_2, p3, p4 = self.decoder(x_, data)
            #----------node-----------#
            node_sample = F.gumbel_softmax(p1, hard=True)
            node_keep_sample = node_sample[:,0]
            node_keep_idx = torch.nonzero(node_keep_sample, as_tuple=False).view(-1,)
            node_edge_index, node_edge_attr = subgraph(node_keep_idx, edge_index, edge_attr, num_nodes=data.num_nodes)
            node_x = x * node_keep_sample.view(-1, 1)

            node_data = copy.deepcopy(data_in)
            node_data.x = node_x
            node_data.edge_index = node_edge_index
            if node_data.edge_attr is not None:
                node_data.edge_attr = node_edge_attr

            #----------mask-----------#
            mask_sample1 = F.gumbel_softmax(p2_1, hard=True)
            mask_node_sample = mask_sample1[:,0]

            mask_sample2 = F.gumbel_softmax(p2_2, hard=True)
            mask_dim_sample = mask_sample2[:,0]
            mask_x = x - (x * mask_node_sample.view(-1, 1)) * mask_dim_sample

            mask_data = copy.deepcopy(data_in)
            mask_data.x = mask_x

            #----------edge-----------#
            edge_sample = F.gumbel_softmax(p3, hard=True)
            edge_keep_sample = edge_sample[:,0].float()

            if edge_weight is not None:
                edge_keep_weight = edge_weight*edge_keep_sample
            else:
                edge_keep_weight = edge_keep_sample

            edge_data = copy.deepcopy(data_in)
            edge_data.edge_weight = edge_keep_weight

            #----------subgraph-----------#
            subgraph_sample = F.gumbel_softmax(p4, hard=True)
            key_sample = subgraph_sample[:,0]
        
            key_idx = torch.nonzero(key_sample, as_tuple=False).view(-1,)
            subset, sub_edge, _, edge_mask = k_hop_subgraph(key_idx, hop, data.edge_index, relabel_nodes=False, flow='target_to_source') 
            subnode_sample = torch.zeros_like(key_sample)
            subnode_sample[subset] = 1
            sub_sample = subnode_sample - key_sample
            x1 = x * key_sample.view(-1, 1)
            x2 = x * sub_sample.view(-1, 1)

            subgraph_data = copy.deepcopy(data_in)
            subgraph_data.edge_index = sub_edge
            subgraph_data.x = x1 + x2
            if subgraph_data.edge_attr is not None:
                subgraph_edge_attr = edge_attr[edge_mask,:]
                subgraph_data.edge_attr = subgraph_edge_attr



            return  node_data, mask_data, edge_data, subgraph_data, node_keep_sample, mask_node_sample, edge_keep_sample, key_sample

