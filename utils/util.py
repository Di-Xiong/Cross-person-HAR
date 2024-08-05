# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import argparse


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append('eval%d_in' % i)
            eval_name_dict['valid'].append('eval%d_out' % i)
        else:
            eval_name_dict['target'].append('eval%d_out' % i)
    return eval_name_dict


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def act_param_init(args):
    args.act_dataset = ['usc','dsads','pamap']
    args.act_people = {'dsads':[[0, 1], [2, 3], [4, 5], [6, 7]]}
    tmp = {'usc': ((6, 1, 200), 12), 'dsads':((45, 1, 125), 19), 'pamap':((27,1,200), 12)}
    
    args.num_classes, args.input_shape = tmp[args.dataset][1], tmp[args.dataset][0]
    return args


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--alpha', type=float,
                        default=0.5, help="DANN dis alpha")
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=32, help="batch_size")
    parser.add_argument('--beta1', type=float, default=0.5, help="Adam")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=100, help='Checkpoint every N steps')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--class_balanced', type=int, default=0)
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dsads')
    parser.add_argument('--data_dir', type=str, default='./data/act/')
    parser.add_argument('--dis_hidden', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--ldmarginlosstype', type=str, default='avg_top_k',
                        choices=['all_top_k', 'worst_top_k', 'avg_top_k'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_decay1', type=float,
                        default=1, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1)
    parser.add_argument('--max_epoch', type=int,
                        default=150, help="max iterations")
    parser.add_argument('--mixupalpha', type=float, default=0.1)
    parser.add_argument('--mixup_ld_margin', type=float, default=10)
    parser.add_argument('--mixupregtype', type=str,
                        default='l-margin', choices=['ld-margin'])
    parser.add_argument('--net', type=str,
                        default='ActNetwork', help="ActNetwork")
    parser.add_argument('--N_WORKERS', type=int, default=1)
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed1', type=int, default=0)
    parser.add_argument('--task', type=str,
                        default="cross_people", choices=['cross_people'])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output', type=str, default="True")
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--wtype', type=str, default='ori',
                        choices=['ori', 'abs', 'fea'])
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--ssc_l',type = float,default=0.5)
    parser.add_argument('--ssc_f',type = float,default=1.2)
    args = parser.parse_args()
    return args


def init_args(args):
    args.steps_per_epoch = 100
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    if args.output =="True":
        args.output ="./results/"+args.dataset+"/"+"seed="+str(args.seed)+"/"+args.algorithm+"/"+str(args.test_envs[0])
    else:
        args.output ="./test_results"+args.dataset+"/"+"seed="+str(args.seed)+"/"+args.algorithm+"/"+str(args.test_envs[0])
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = act_param_init(args)
    return args
