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
from trans_wgine_conv import transWGINEConv
from pre_embed import Embed

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINE_WeightEncoder(torch.nn.Module):
   def __init__(self, emb_dim):
       super().__init__()

       self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
       self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

       torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
       torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

       self.conv1 = transWGINEConv(emb_dim)
       self.bn1 = torch.nn.BatchNorm1d(emb_dim)

       self.conv2 = transWGINEConv(emb_dim)
       self.bn2 = torch.nn.BatchNorm1d(emb_dim)

       for m in self.modules():
           if isinstance(m, torch.nn.Linear):
               nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
               if m.bias is not None:
                   fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                   bound = 1 / math.sqrt(fan_in)
                   nn.init.uniform_(m.bias, -bound, bound)
           elif isinstance(m, torch.nn.BatchNorm1d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
       
   def forward(self, data):        
       x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
       x = self.x_embedding1(x[:,0].long()) + self.x_embedding2(x[:,1].long())

       x = F.relu(self.conv1(x, edge_index, edge_attr))
       x = self.bn1(x)
       
       x = F.relu(self.conv2(x, edge_index, edge_attr))
       x = self.bn2(x)
       return x
   
class View_expert(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_expert = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.mask_expert1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.edge_expert = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.subgraph_expert = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
    
    def forward(self, x_, data):
        p1 = self.node_expert(x_)

        p2_1 = self.mask_expert1(x_)

        node_attr = x_[data.edge_index]
        edge_embed = torch.cat([node_attr[0],node_attr[1]],1)
        p3 = self.edge_expert(edge_embed)

        p4 = self.subgraph_expert(x_)

        return p1, p2_1, p3, p4
    
class CompositionalViewGenerator(VGAE):
        def __init__(self, dataset, hidden_dim):
            encoder = GINE_WeightEncoder(hidden_dim)
            decoder = View_expert(hidden_dim)
            super().__init__(encoder=encoder, decoder=decoder)
                    
        def forward(self, data_in, requires_grad, hop):
            data = copy.deepcopy(data_in)
            
            x, edge_index, edge_weight, edge_attr = data.x, data.edge_index, data.edge_weight, data.edge_attr
            x = x.float()
            x.requires_grad = requires_grad

            x_ = self.encoder(data)
            p1, p2_1, p3, p4 = self.decoder(x_, data)
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
            mask_idx = mask_node_sample.bool()
            token = torch.zeros_like(x,device=x.device)
            token[mask_idx,:] = torch.tensor([num_atom_type-1,0],dtype=torch.float,device=token.device)
            mask_x = x - (x * mask_node_sample.view(-1, 1)) + token

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



            return  node_data, mask_data, edge_data, subgraph_data, node_keep_sample, mask_node_sample, edge_keep_sample, key_sample, subnode_sample

