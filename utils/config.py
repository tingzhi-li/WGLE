import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MINIMUM = 1e-16
SEED = 42

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora') # 'Cora', 'DBLP', 'Photo', 'Computers', 'CS', 'Physics'
    parser.add_argument('--dataset_path', type=str, default='./datasets/')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_val_test', type=list, default=[0.7, 0.2, 0.1])

    parser.add_argument('--model', type=str, default='GCNv2') # 'GCNv2', 'SSG', 'SAGE', 'ARMA', 'GEN', 'GTF'
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--hidden_channels', type=list, default=[600, 400, 200])
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--trigger_epochs', type=int, default=201)
    parser.add_argument('--trigger_lr', type=float, default=1e-4)
    parser.add_argument('--wm_lr', type=float, default=1e-4)

    parser.add_argument('--setting', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=501)
    parser.add_argument('--wm_epochs', type=int, default=501)
    parser.add_argument('--n_wm', type=int, default=200)
    parser.add_argument('--coe', type=float, default=1.0)
    parser.add_argument('--model_num', type=int, default=100)
    parser.add_argument('--results_path', type=str, default='./results/')
    return parser.parse_args()