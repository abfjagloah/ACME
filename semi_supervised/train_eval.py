import logging
import sys
import time
import copy
from sklearn.model_selection import StratifiedKFold
import random
import argparse
import os

import torch
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from torch.utils.data import random_split
import itertools

from pre_GNN import pre_gnn
from view_generator import CompositionalViewGenerator

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

def test_pre_model(model,loader, device):
    model.eval()
    correct = 0
    for data in loader:   
        data = data.to(device)                         # 批遍历测试集数据集。
        out  = model(data) # 一次前向传播
        pred = out.argmax(dim=1)                         # 使用概率最高的类别
        correct += int((pred == data.y).sum())           # 检查真实标签
    return correct / len(loader.dataset)

def train_pre_model(model, pre_dataset, args, device):
    print('Train Pre_GNN')
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
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            loss_all += loss.item()
            loss.backward()
            pre_optimizer.step()
        train_acc =  test_pre_model(model,train_loader,device)
        test_acc =  test_pre_model(model,test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), './pth/pre_encoder_{}.pth'.format(args.dataset))
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print('Pre_GNN Done!')

def benchmark_exp(device, logger, dataset, model_func, 
                 folds, epochs, batch_size,
                 lr, lr_decay_factor, lr_decay_step_size, weight_decay, 
                 epoch_select, with_eval_mode=True, semi_split=None):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        # train_dataset = dataset[train_idx]
        semi_dataset = dataset[semi_idx]
        # val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]

        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Pre-training Classifier...")
        for epoch in range(1, epochs + 1):
            train_loss = train_cls(model, optimizer, semi_loader, device)
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            if epoch % 10 == 0:
                logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                    'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss, 
                                                                train_acc, test_acc))

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)

    _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
    selected_epoch = selected_epoch.repeat(folds)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    # best_epoch = test_acc.max(dim=1)[1]
    # test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(selected_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

def cl_pretrain(args, device, logger, dataset, pre_dataset, model_func, epochs, batch_size,
            lr, weight_decay, epoch_select):
    
    assert epoch_select in ['val_max', 'test_max'], epoch_select

    pre_encoder = model_func(dataset)
    pre_model = pre_gnn(pre_encoder, dataset.num_features, args.hidden, pre_dataset)
    pre_model = pre_model.to(device)
    if not os.path.isfile('./pth/pre_encoder_{}.pth'.format(args.dataset)):
        train_pre_model(pre_model, pre_dataset, args, device)
    pre_model.load_state_dict(torch.load('./pth/pre_encoder_{}.pth'.format(args.dataset),map_location=device))
    pre_model.eval()
    
    train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

    model = model_func(dataset)
    view_gen = CompositionalViewGenerator(dataset, args.hidden, 2, batch_size)
    model.to(device)
    view_gen = view_gen.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    view_optimizer = Adam(view_gen.parameters(), lr=lr, weight_decay=weight_decay)
    
    logger.info("*" * 50)
    logger.info("Pre-Training View Generator and Encoder")

    for epoch in range(1, epochs + 1):
        loss, cl_loss = train_cl_with_fix_node_weight_view_gen(view_gen, view_optimizer, model, 
                                            optimizer, pre_model, train_loader, device)
        logger.info('Epoch: {:03d}, Loss: {:.4f}, CL Loss: {:.4f}'.format(epoch, loss, cl_loss))
    torch.save(model.state_dict(), './models/pre_model_{}.pth'.format(args.dataset))

def cl_finetune(args, device, logger, dataset, pre_dataset, model_func, folds, epochs, batch_size,
            lr, lr_decay_factor, lr_decay_step_size, weight_decay, epoch_select,
            with_eval_mode=True, semi_split=None):
    
    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        test_dataset = dataset[test_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Semi size: %d" % len(semi_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)
        model.load_state_dict(torch.load('./models/pre_model_{}.pth'.format(args.dataset), map_location=device))
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Training Encoder and Classifier...")
        best_test_acc = 0
        after_train_view_snapshot = None
                                                
        for epoch in range(1, epochs + 1):
            train_loss = train_cls(model, optimizer, semi_loader, device)

            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            if epoch % 10 == 0:
                logger.info('Epoch: {:03d}, Train View Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss, train_acc, test_acc))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)

    _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
    selected_epoch = selected_epoch.repeat(folds)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    # best_epoch = test_acc.max(dim=1)[1]
    # test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(selected_epoch))

    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

def train_cl_with_fix_node_weight_view_gen(view_gen, view_optimizer, 
                                        model, optimizer, info_model, loader, device):
    model.train()
    view_gen.train()

    loss_all = 0
    cl_loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()
        view_optimizer.zero_grad()
        
        data = data.to(device)
        node_data, mask_data, edge_data, subgraph_data, nk_sample, mn_sample, ek_sample,key_sample = view_gen(data, True, 2)

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
        for b_id in range(loader.batch_size):
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

        info_list = []
        out_list = []
        for view in view_list:
            out_list.append(model.forward_cl(view))
            view_info = info_model.get_embedding(view)
            info_list.append(view_info)
        
        comb1 = itertools.combinations(range(len(info_list)), 2)
        sim_loss = 0
        cl_loss = 0
        pdist = nn.CosineSimilarity(dim=0, eps=1e-6)
        for c in comb1:
            sim_loss -= loss_cl(info_list[c[0]],info_list[c[1]])
            cl_loss += loss_cl(out_list[c[0]],out_list[c[1]])

        regularize = 0
        for i in range(len(reg)):
            regularize += 1000 * (int(reg[i] >= 0.6) * reg[i])
        loss = 1.5 * cl_loss + sim_loss + regularize + reg[3]
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        # sim_loss_all += sim_loss.item() * data.num_graphs
        cl_loss_all += cl_loss.item() * data.num_graphs

        total_graphs += data.num_graphs
        view_optimizer.step()
        optimizer.step()
    
    loss_all /= total_graphs
    # sim_loss_all /= total_graphs
    cl_loss_all /= total_graphs
    return loss_all, cl_loss_all

def cl_k_fold(dataset, folds, epoch_select, semi_split):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []

    semi_indices = []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    dataset_size = len(dataset)
    semi_size = int(dataset_size * semi_split / 100)

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indice = torch.nonzero(train_mask, as_tuple=False).view(-1)
        train_indices.append(train_indice)
        train_size = train_indice.shape[0]
        select_idx = torch.randperm(train_size)[:semi_size]
        semi_indice = train_indice[select_idx]
        semi_indices.append(semi_indice)
        
    return train_indices, test_indices, val_indices, semi_indices

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def train_cls(model, optimizer, loader, device):
    model.train()

    loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()    
        data = data.to(device)
        output = model(data)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        optimizer.step()
    
    loss_all /= total_graphs
    return loss_all

@torch.no_grad()
def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
