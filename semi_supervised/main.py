import re
import sys
import argparse
import time
import random
import numpy as np
import os
import shutil
import glob
import logging
from itertools import product
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join('..')))
from train_eval import cl_pretrain, cl_finetune, benchmark_exp
from res_gcn import ResGCN
from datasets import get_dataset



str2bool = lambda x: x.lower() == "true"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--exp', type=str, default="benchmark")
parser.add_argument('--data_root', type=str, default="data")
parser.add_argument('--dataset', type=str, default="")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save', type=str, default="EXP")
parser.add_argument('--n_fold', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=300)
parser.add_argument('--epoch_select', type=str, default='test_max')
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=3)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)
parser.add_argument('--with_eval_mode', type=str2bool, default=True)
parser.add_argument('--semi_split', type=int, default=10, help='1/percent of semi training data')
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc = info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
            fold, epoch, train_acc, test_acc))
    sys.stdout.flush()
def create_n_filter_triples(datasets, feat_strs, nets, gfn_add_ak3=False,
                            gfn_reall=True, reddit_odeg10=False,
                            dd_odeg10_ak1=False):
    triples = [(d, f, n) for d, f, n in product(datasets, feat_strs, nets)]
    triples_filtered = []
    for dataset, feat_str, net in triples:
        # Add ak3 for GFN.
        if gfn_add_ak3 and 'GFN' in net:
            feat_str += '+ak3'
        # Remove edges for GFN.
        if gfn_reall and 'GFN' in net:
            feat_str += '+reall'
        # Replace degree feats for REDDIT datasets (less redundancy, faster).
        if reddit_odeg10 and dataset in [
                'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        # Replace degree and akx feats for dd (less redundancy, faster).
        if dd_odeg10_ak1 and dataset in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')
        triples_filtered.append((dataset, feat_str, net))
    return triples_filtered

def get_model_with_default_configs(model_name,
                                   num_feat_layers=args.n_layers_feat,
                                   num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc,
                                   residual=args.skip_connection,
                                   hidden=args.hidden):
    # More default settings.
    res_branch = args.res_branch
    global_pool = args.global_pool
    dropout = args.dropout
    edge_norm = args.edge_norm

    # modify default architecture when needed
    if model_name.find('_') > 0:
        num_conv_layers_ = re.findall('_conv(\d+)', model_name)
        if len(num_conv_layers_) == 1:
            num_conv_layers = int(num_conv_layers_[0])
            print('[INFO] num_conv_layers set to {} as in {}'.format(
                num_conv_layers, model_name))
        num_fc_layers_ = re.findall('_fc(\d+)', model_name)
        if len(num_fc_layers_) == 1:
            num_fc_layers = int(num_fc_layers_[0])
            print('[INFO] num_fc_layers set to {} as in {}'.format(
                num_fc_layers, model_name))
        residual_ = re.findall('_res(\d+)', model_name)
        if len(residual_) == 1:
            residual = bool(int(residual_[0]))
            print('[INFO] residual set to {} as in {}'.format(
                residual, model_name))
        gating = re.findall('_gating', model_name)
        if len(gating) == 1:
            global_pool += "_gating"
            print('[INFO] add gating to global_pool {} as in {}'.format(
                global_pool, model_name))
        dropout_ = re.findall('_drop([\.\d]+)', model_name)
        if len(dropout_) == 1:
            dropout = float(dropout_[0])
            print('[INFO] dropout set to {} as in {}'.format(
                dropout, model_name))
        hidden_ = re.findall('_dim(\d+)', model_name)
        if len(hidden_) == 1:
            hidden = int(hidden_[0])
            print('[INFO] hidden set to {} as in {}'.format(
                hidden, model_name))

    if model_name.startswith('ResGFN'):
        collapse = True if 'flat' in model_name else False
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=True, collapse=collapse,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    elif model_name.startswith('ResGCN'):
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    else:
        raise ValueError("Unknown model {}".format(model_name))
    return foo

def run_benchmark_exp(args, device, logger):
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    dataset_feat_net_triples = create_n_filter_triples(datasets, feat_strs, nets, gfn_add_ak3=True, reddit_odeg10=True, dd_odeg10_ak1=True)
    
    results = []
    exp_nums = len(dataset_feat_net_triples)

    logger.info("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
    sys.stdout.flush()

    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('-----\n{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)

        model_func = get_model_with_default_configs(net)
        train_acc, acc, std, duration = benchmark_exp(
            device,
            logger,
            dataset,
            model_func,
            folds=args.n_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            epoch_select=args.epoch_select,
            with_eval_mode=args.with_eval_mode,
            semi_split=args.semi_split)

        summary1 = 'data={}, model={}, feat={}, eval={}'.format(
            dataset_name, net, feat_str, args.epoch_select)
        summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
            train_acc*100, acc*100, std*100, round(duration, 2))
        results += ['{}: {}, {}'.format('fin-result', summary1, summary2)]
        logger.info('{}: {}, {}'.format('mid-result', summary1, summary2))
    logger.info('-----\n{}'.format('\n'.join(results)))
    benchmark_log_path = './exp/benchmark_results.txt'
    with open(benchmark_log_path, 'a+') as f:
        f.write('{} {}\n'.format(results,time.strftime("%Y-%m-%d %H:%M:%S")))
        # f.write(results)

def run_cl_pretrain(args, device, logger):
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    dataset_feat_net_triples = create_n_filter_triples(datasets, feat_strs, nets,
                                        gfn_add_ak3=True,
                                        reddit_odeg10=True,
                                        dd_odeg10_ak1=True)

    exp_nums = len(dataset_feat_net_triples)

    logger.info("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
    sys.stdout.flush()

    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('-----\n{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
        pre_dataset = get_dataset(
            'AIDS', sparse=True, feat_str=feat_str, root=args.data_root)
        
        model_func = get_model_with_default_configs(net)
        cl_pretrain(
            args,
            device,
            logger,
            dataset,
            pre_dataset,
            model_func,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=0,
            epoch_select=args.epoch_select)

def run_cl_finetune(args, device, logger):
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    dataset_feat_net_triples = create_n_filter_triples(datasets, feat_strs, nets,
                                        gfn_add_ak3=True,
                                        reddit_odeg10=True,
                                        dd_odeg10_ak1=True)

    results = []
    exp_nums = len(dataset_feat_net_triples)

    logger.info("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
    sys.stdout.flush()

    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        logger.info('-----\n{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
        pre_dataset = get_dataset(
            'AIDS', sparse=True, feat_str=feat_str, root=args.data_root)
        
        model_func = get_model_with_default_configs(net)
        train_acc, acc, std, duration = cl_finetune(
            args,
            device,
            logger,
            dataset,
            pre_dataset,
            model_func,
            folds=args.n_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            epoch_select=args.epoch_select,
            with_eval_mode=args.with_eval_mode,
            semi_split=args.semi_split)

        summary1 = 'data={}, seed={}, skip={}, branch={}, pool={}, drop={}, n_feat={}, n_conv={}, n_fc={}, hidden={}, pre_epoch={}, pre_batch={}, pre_lr={}, batch={}, lr={}'.format(
            dataset_name, args.seed, args.skip_connection, args.res_branch, args.global_pool, args.dropout, args.n_layers_feat, args.n_layers_conv,\
                args.n_layers_fc, args.hidden, args.pre_epoch, args.pre_batch_size, args.pre_lr, args.batch_size, args.lr)
        summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
            train_acc*100, acc*100, std*100, round(duration, 2))
        results += ['{}: {}, {}'.format('fin-result', summary1, summary2)]
        logger.info('{}: {}, {}'.format('mid-result', summary1, summary2))
    logger.info('-----\n{}'.format('\n'.join(results)))
    joint_log_path = './exp/{}.txt'.format(args.save)
    with open(joint_log_path, 'a+') as f:
        f.write('{}\n'.format(results))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))
    
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


if __name__ == '__main__':
    set_seed(args.seed)

    device_id = 'cuda:%d' % (args.gpu)
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

    save_path = '{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
    save_path = os.path.join('exp', args.exp, save_path)
    create_exp_dir(save_path, glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.info(args) 

    if args.exp == 'benchmark':
        run_benchmark_exp(args, device, logger)
    elif args.exp == 'cl_pretrain':
        run_cl_pretrain(args, device, logger)
    elif args.exp == 'cl_finetune':
        run_cl_finetune(args, device, logger)
    else:
        raise ValueError('Unknown exp {} to run'.format(args.exp))
