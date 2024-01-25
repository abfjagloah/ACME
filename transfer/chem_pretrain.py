import argparse
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import time
import shutil
import glob
import logging
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from chem_splitters import scaffold_split, random_split, random_scaffold_split

from chem_loader import MoleculeDataset_aug
from chem_loader import MoleculeDataset
from chem_model import GNN_CL
from trans_view_generator import CompositionalViewGenerator
import itertools
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join('..')))

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset_root', type=str, default = 'dataset', help='root directory of dataset')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--pre_dataset', type=str, default = 'esol', help='root directory of pre_dataset.')
    parser.add_argument('--exp', type=str, default = 'chem', help='')
    parser.add_argument('--save', type=str, default = '', help='')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=321, help = "Seed for splitting dataset.")
    parser.add_argument('-hop', type=int, default=2, help = "Hop for subgraph.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'none')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'none')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)        
        os.mkdir(os.path.join(path, 'model'))

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

class graphcl(nn.Module):

    def __init__(self, gnn, dataset):
        super().__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        # self.cls_head = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(300, dataset.num_classes))
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.graph_pred_linear = torch.nn.Linear(300, 1)
        self.fusion_layer = nn.Sequential(nn.Linear(4, 4), nn.ReLU(inplace=True), nn.Linear(4, 4))
        # self.cls_head = nn.Sequential(nn.Linear(300, 1))

    def forward(self, data, sample=None):
        if sample is None:
            sample = torch.ones(data.x.shape[0],device=data.x.device)
        x, batch = data.x, data.batch
        x = self.gnn(data, sample)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def pre_forward(self, data, sample=None):
        if sample is None:
            sample = torch.ones(data.x.shape[0],device=data.x.device)
        x, batch = data.x, data.batch
        x = self.gnn(data, sample)
        x = self.pool(x, batch)
        # out = self.projection_head(x)
        out = self.graph_pred_linear(x)
        return x, out

def loss_cl(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean()

    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss

def train_node_view_cl_with_sim(view_gen, view_optimizer, model, optimizer, info_model, data_loader, device, args, logger):
    model.train()
    view_gen.train()

    cl_loss_accum = 0
    train_loss_accum = 0
    total_graphs = 0

    with tqdm(data_loader, ncols=100, ascii=' >') as t:
        for data in data_loader:
            data = data.to(device)   
            optimizer.zero_grad()
            view_optimizer.zero_grad()

            node_data, mask_data, edge_data, subgraph_data, nk_sample, mn_sample, ek_sample,key_sample, subnode_sample = view_gen(data, True, args.hop)
            view_list = [data, node_data, mask_data, edge_data, subgraph_data]

            row, col = data.edge_index
            edge_batch = data.batch[row]
            batch = data.batch

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            uni_n, batch_num = batch.unique(return_counts=True)
            sum_ke = scatter( ek_sample, edge_batch, reduce="sum")
            sum_kn = scatter( nk_sample, batch, reduce="sum")
            sum_mn = scatter( mn_sample, batch, reduce="sum")
            sum_key = scatter( key_sample, batch, reduce="sum")

            reg = [[]for i in range(4)]
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    num_nodes = batch_num[uni_n.tolist().index(b_id)]
                    reg[0].append(1.0-sum_kn[b_id] / num_nodes)
                    reg[1].append(sum_mn[b_id] / num_nodes)
                    reg[2].append(1.0-sum_ke[b_id] / num_edges)
                    reg[3].append(1.0-sum_key[b_id] / num_nodes)
                else:
                    # means no edges in that graph. So don't include.
                    pass

            reg = [torch.stack(reg[i]) for i in range(4)]
            reg = [reg[i].mean() for i in range(4)]

            sample = torch.ones_like(nk_sample, device=nk_sample.device)
            sample_list = [sample, nk_sample, sample, sample, subnode_sample]

            info_list = []
            out_list = []
            for i in range(len(view_list)):
                view_info, _ = info_model.pre_forward(view_list[i], sample_list[i])
                info_list.append(view_info)
                out_list.append(model(view_list[i], sample_list[i]))
            
            comb = itertools.combinations(range(len(info_list)), 2)
            sim_loss = 0
            cl_loss = 0
            pdist = nn.CosineSimilarity(dim=0, eps=1e-6)
            for c in comb:
                sim_loss -= loss_cl(info_list[c[0]],info_list[c[1]])           
                cl_loss += loss_cl(out_list[c[0]], out_list[c[1]])
            
            regularize = 0
            for i in range(len(reg)):
                regularize += 1000 * (int(reg[i] >= 0.6) * reg[i]) - 1000 * (int(reg[i] <= 0.05) * reg[i])#0.01 * reg[i]

            loss = cl_loss + sim_loss + regularize
            loss.backward()

            optimizer.step()
            view_optimizer.step()

            train_loss_accum += loss.item() * data.num_graphs
            cl_loss_accum += cl_loss.item() * data.num_graphs
            total_graphs += data.num_graphs
            postfix_str = 'sim_loss: {:.04f}, cl_loss: {:.04f}, reg: {:.04f}'.format(sim_loss, cl_loss, regularize)

            t.set_postfix_str(postfix_str)
            t.update()

    return train_loss_accum / total_graphs, cl_loss_accum / total_graphs

def test_pre_model(model,loader):
    model.eval()
    mse = 0
    criterion = torch.nn.L1Loss()
    for data in loader:
        data = data.to(args.device)                         # 批遍历测试集数据集。
        _, out = model.pre_forward(data)
                              # huigui
        mse += criterion(out, data.y.to(torch.float32))           # 
    return mse / len(loader)

def train_pre_model(model, pre_dataset):
    print('Train Pre_GNN')
    smiles_list = pd.read_csv(os.path.join(args.dataset_root,args.pre_dataset, 'processed/smiles.csv'), header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(pre_dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    pre_optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
    criterion = torch.nn.L1Loss()

    best_test_mse = 9999
    for epoch in range(1, 51):
        model.train()
        loss_all = 0
        for data in train_loader:
            pre_optimizer.zero_grad()
            data = data.to(args.device)
            _, out = model.pre_forward(data)
            loss = criterion(out, data.y.to(torch.float32))#regression
            loss_all += loss.item()
            loss.backward()
            pre_optimizer.step()
        train_mse =  test_pre_model(model,train_loader)
        test_mse =  test_pre_model(model,test_loader)
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            torch.save(model.state_dict(), './pth/pre_encoder.pth')
        print(f'Epoch: {epoch:03d}, Loss: {loss_all:.4f}, Train mse: {train_mse:.4f}, Test mse: {test_mse:.4f}')
    print('Pre_GNN Done!')

                           
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    args.save = '{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('transfer_exp', args.exp, args.save)
    create_exp_dir(args.save, glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.info(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    #set up dataset
    dataset = MoleculeDataset(os.path.join(args.dataset_root, args.dataset), dataset=args.dataset)
    logger.info(dataset)
    dataset = dataset.shuffle()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=True)

    pre_dataset = MoleculeDataset(os.path.join(args.dataset_root, args.pre_dataset), dataset=args.pre_dataset)
    logger.info(pre_dataset)
    pre_dataset = pre_dataset.shuffle()

    gnn = GNN_CL(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model = graphcl(gnn, dataset)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    view_gen = CompositionalViewGenerator(dataset, args.emb_dim)
    view_gen = view_gen.to(device)
    view_optimizer = optim.Adam( view_gen.parameters(), lr=args.lr, weight_decay=args.decay)

    pre_gnn = GNN_CL(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    pre_encoder = graphcl(pre_gnn, pre_dataset)
    pre_encoder = pre_encoder.to(device)
    if not os.path.isfile('./pth/pre_encoder.pth'):
        train_pre_model(pre_encoder, pre_dataset)
    pre_encoder.load_state_dict(torch.load('./pth/pre_encoder.pth'))
    pre_encoder.eval()
    
    logger.info("Exp Dir: {}".format(args.save))

    for epoch in range(1, args.epochs+1):    
        train_loss, cl_loss = train_node_view_cl_with_sim(view_gen, view_optimizer, model, optimizer, pre_encoder, train_loader, device, args, logger)
        logger.info('Epoch: {}, Train Loss: {:.4f}, cl_loss: {:.4f}'.format(epoch, train_loss, cl_loss))

        if epoch % 10 == 0:
            model_name = "cl_model_{}.pth".format(epoch)
            model_path = os.path.join(args.save, 'model', model_name)
            torch.save(gnn.state_dict(), model_path)
            torch.save(gnn.state_dict(), './pretrain_models/zinc321/'+model_name)