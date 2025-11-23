import torch
import time
import argparse

MINIMUM = 1e-8  # A small epsilon value to avoid division by zero errors
SEED = 42       # Fixed random seed for reproducible dataset splitting; do not modify after initial use

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora') # 'Cora', 'CiteSeer', 'DBLP', 'Photo', 'CS', 'Physics'
    parser.add_argument('--paradigm', type=str, default='transductive') # 'inductive', 'transductive'
    parser.add_argument('--dataset_path', type=str, default='./datasets/')
    parser.add_argument('--train_val_test', nargs=3, type=float, default=[0.4, 0.3, 0.3])

    parser.add_argument('--model', type=str, default='SAGE') # 'GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF'
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--hidden_channels', nargs=3, type=int, default=[800, 200, 50])
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--trigger_epochs', type=int, default=201)
    parser.add_argument('--trigger_lr', type=float, default=1e-4)
    parser.add_argument('--wm_lr', type=float, default=5e-5)

    parser.add_argument('--setting', type=int, default=1) # 1, 2
    parser.add_argument('--max_epochs', type=int, default=201)
    parser.add_argument('--wm_epochs', type=int, default=501)
    parser.add_argument('--n_wm', type=int, default=64)
    parser.add_argument('--coe', type=float, default=5.0)
    parser.add_argument('--model_num', type=int, default=100)
    parser.add_argument('--results_path', type=str, default='./results/')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args
