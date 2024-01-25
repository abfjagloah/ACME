import os
import numpy as np
import random
import sys
import argparse
import time
import logging
import shutil
import glob
sys.path.append(os.path.abspath(os.path.join('..')))
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from torch_scatter import scatter
import itertools
from torch.utils.data import random_split
from us_model import simclr,pre_gnn
from view_generator import CompositionalViewGenerator
from us_evaluate_embedding import evaluate_embedding
from datasets import get_dataset

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--dataset', dest='dataset', help='Dataset')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int, default=2, help='Number of layers for GCL_Encoder')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=128, help='')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save', type=str, default = 'debug', help='')
    parser.add_argument('--batch_size', type=int, default = 256, help='')
    parser.add_argument('--epochs', type=int, default = 50, help='')
    parser.add_argument('--device', type=str, default='cpu',help='')
    parser.add_argument('--hop', type=int, default=2, help='')
    parser.add_argument('--pool', type=str, default='sum', help='')
    parser.add_argument('--tau', type=float, default=0.5, help='tau for CL_loss')
    return parser.parse_args()

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
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

def eval_acc(model, data_loader, device, pool):
    with torch.no_grad():
        model.eval()
        emb, y = model.encoder.get_embeddings(data_loader, pool, device)
        acc, std= evaluate_embedding(emb, y)
        return acc, std

def loss_cl(x1, x2):
    T = args.tau
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

def train_one_epoch(model, view_gen, info_model, optimizer, view_optimizer, data_loader, device, hop):
    cl_loss_all = 0
    sim_loss_all = 0
    total_graphs = 0
    loss_all = 0

    model.train()
    view_gen.train()

    for data in data_loader:
        optimizer.zero_grad()
        view_optimizer.zero_grad()
        data = data.to(device)

        node_data, mask_data, edge_data, subgraph_data, nk_sample, mn_sample, ek_sample,key_sample = view_gen(data, True, hop)

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
                reg[3].append(sum_key[b_id] / num_nodes)
            else:
                # means no edges in that graph. So don't include.
                pass

        reg = [torch.stack(reg[i]) for i in range(4)]
        reg = [reg[i].mean() for i in range(4)]

        info_list = []
        out_list = []
        for view in view_list:
            view_info = info_model.get_embedding(view, args.pool)
            info_list.append(view_info)
            out_list.append(model(view, args.pool))
        
        comb1 = itertools.combinations(range(len(info_list)), 2)
        sim_loss = 0
        cl_loss = 0
        for c in comb1:
            sim_loss -= loss_cl(info_list[c[0]],info_list[c[1]])
            cl_loss += loss_cl(out_list[c[0]],out_list[c[1]])

        regularize = 0
        for i in range(len(reg)-1):
            regularize +=  int(reg[i] >= 0.6) * reg[i]
        
        loss = cl_loss + sim_loss + regularize + reg[3]

        loss_all = loss.item() * data.num_graphs
        cl_loss_all += cl_loss.item() * data.num_graphs
        sim_loss_all += sim_loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        loss.backward()

        optimizer.step()
        view_optimizer.step()

    loss_all /= total_graphs
    cl_loss_all /= total_graphs
    sim_loss_all /= total_graphs
    
    return loss_all, cl_loss_all, sim_loss_all

def test_pre_model(model,loader):
    model.eval()
    correct = 0
    for data in loader:   
        data = data.to(args.device)                         
        out  = model(data, args.pool) 
        pred = out.argmax(dim=1)                         
        correct += int((pred == data.y).sum())          
    return correct / len(loader.dataset)

def train_pre_model(model, pre_dataset):
    logger.info('Train Pre_GNN')
    logger.info('=============')
    train_size = int(len(pre_dataset) * 0.8)
    test_size = len(pre_dataset) - train_size
    train_dataset, test_dataset = random_split(pre_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    pre_optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_test_acc = 0
    for epoch in range(1, 51):
        model.train()
        loss_all = 0
        for data in train_loader:
            pre_optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data, args.pool)
            loss = criterion(out, data.y)
            loss_all += loss.item()
            loss.backward()
            pre_optimizer.step()
        train_acc =  test_pre_model(model,train_loader)
        test_acc =  test_pre_model(model,test_loader)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), '{}/pre_encoder_{}.pth'.format(pth_path,args.dataset))
        logger.info(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    logger.info('Pre_GNN Done!')
    logger.info('=============')

def train(args):
    set_seed(args.seed)
    device = args.device
    epochs = args.epochs
    log_interval = 5
    batch_size = args.batch_size
    lr = args.lr

    dataset = get_dataset(args.dataset, sparse=True, feat_str='deg+odeg10', root='./data') 
    pre_dataset = get_dataset('AIDS', sparse=True, feat_str='deg+odeg10', root='./data')
    pre_dataset = pre_dataset.shuffle()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    data_eval_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    
    model = simclr(dataset, args.hidden_dim, args.num_gc_layers)
    view_gen = CompositionalViewGenerator(dataset, args.hidden_dim, 2, batch_size)
    pre_encoder = simclr(dataset, args.hidden_dim, args.num_gc_layers)
    pre_model = pre_gnn(pre_encoder, dataset.num_features, args.hidden_dim*args.num_gc_layers, pre_dataset)
    model = model.to(device)
    view_gen = view_gen.to(device)
    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    view_optimizer = torch.optim.Adam(view_gen.parameters(), lr=lr)

    try:
        pre_model.load_state_dict(torch.load('{}/pre_encoder_{}.pth'.format(pth_path,args.dataset)))
    except:
        train_pre_model(pre_model, pre_dataset)
        pre_model.load_state_dict(torch.load('{}/pre_encoder_{}.pth'.format(pth_path,args.dataset)))
    pre_model.eval()

    best_test_acc = 0
    best_test_std = 0
    best_epoch = 0
    test_acc = 0
    logger.info('Seed: {}'.format(args.seed))

    for epoch in range(1, epochs+1):

        loss, cl_loss, sim_loss = train_one_epoch(model, view_gen, pre_model, optimizer, view_optimizer, data_loader, device, args.hop)

        logger.info("Epoch:{}, loss:{:.4f}, sim_loss:{:.4f}, cl_loss:{:.4f}".format(epoch,loss,sim_loss,cl_loss))
        if epoch % log_interval == 0:
                logger.info("Evaluating embedding...")
                test_acc, test_std = eval_acc(model, data_eval_loader, device, args.pool)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_test_std = test_std
                    best_epoch = epoch
                logger.info('Epoch: {}, Test Acc: {:.2f} ± {:.2f}'.format(epoch, test_acc*100, test_std*100))
    logger.info('=====================')
    logger.info('Best epoch:{}, Acc: {:.2f}, {:.2f}\n'.format(best_epoch, best_test_acc*100, best_test_std*100))

    return best_test_acc, best_test_std, best_epoch

if __name__ == '__main__':
    args = arg_parse()
    if(args.gpu == -1): 
        args.device=torch.device('cpu')
    else:
        device_id = 'cuda:%d' % (args.gpu)
        args.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

    joint_log_name = 'joint_log_{}.txt'.format(args.save)
    save_name = args.save
    joint_log_dir = os.path.join('unsupervised_exp', save_name)
    joint_log_path = os.path.join(joint_log_dir, joint_log_name)
    pth_path = './pth/{}_pth'.format(save_name)
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    if not os.path.exists(joint_log_dir):
        os.makedirs(joint_log_dir)
    savepath = '{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H:%M:%S"))
    savepath = os.path.join('unsupervised_exp', save_name, args.dataset, savepath)
    create_exp_dir(savepath)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(savepath, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.info('================')
    logger.info('dataset: {}'.format(args.dataset))
    logger.info('epochs: {}'.format(args.epochs))
    logger.info('lr: {}'.format(args.lr))
    logger.info('batch_size: {}'.format(args.batch_size))
    logger.info('hidden_dim: {}'.format(args.hidden_dim))
    logger.info('hop: {}'.format(args.hop))
    logger.info('num_gc_layers: {}'.format(args.num_gc_layers))
    logger.info('device: {}'.format(args.device))
    logger.info('seed: {}'.format(args.seed))
    logger.info('save: {}'.format(args.save))
    logger.info('tau: {}'.format(args.tau))
    logger.info('================')
    with open(joint_log_path, 'a+') as f:
        f.write('{}, num_layers:{}, hidden_dim:{}, batch_size:{}, lr:{}, tau:{}, hop:{} '.format(
            args.dataset,args.num_gc_layers,args.hidden_dim,args.batch_size,args.lr, args.tau, args.hop))
    
    accs = []
    for seed in (123,132,231,312,321):
        args.seed = seed
        acc,std, epoch = train(args)
        accs.append(acc)

    with open(joint_log_path, 'a+') as f:
        f.write('Avg acc: {:.2f} ± {:.2f}\n'.format(np.mean(accs)*100, np.std(accs)*100))
    logger.info('==========================================================')
    logger.info('Avg_Acc: {:.2f} ± {:.2f}'.format(np.mean(accs)*100, np.std(accs)*100))
    logger.info('==========================================================\n')
    logger.removeHandler(fh)

