from utils.dataload import *
from watermark.robust import *
from utils.utils import *

import os
import csv
import time
import copy
import torch
import numpy as np
import torch.nn.functional as F

from utils.dataload import load_data
from watermark.robust import model_pruning, model_extract
from utils.utils import test, train
from utils.config import parse_args
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from utils.models import GCNv2, SSG, SAGE, ARMA, GEN, GTF


def WGB(r=0.15, pattern=0.42, poison_label=2, args=None):
    datasets = ['Cora', 'DBLP', 'Photo', 'Computers', 'CS', 'Physics']
    models = [GCNv2, SSG, SAGE, ARMA, GEN, GTF]
    headers = [
        'Train Acc', 'Val Acc', 'Test Acc', 'Time', 'Mc BAR', 'Mb BAR',
        'Misuse backdoor', 'Use backdoor',
        'SAGE Test Acc', 'SAGE BAR', 'SSG Test Acc', 'SSG BAR'
    ]
    headers += [f'{i}%test_acc' for i in range(0, 101, 10)] + [f'{i}%bar' for i in range(0, 101, 10)]
    headers += [f'{i}test_acc' for i in range(200)] + [f'{i}bar' for i in range(200)]

    for idx, dataset in enumerate(datasets):
        args.dataset = dataset
        filename = f'{dataset}_WGB.csv'

        if not os.path.isfile(filename):
            with open(filename, mode='w', newline='') as file:
                csv.writer(file).writerow(headers)

        train_data, val_data, test_data, num_features, num_labels = load_data(args)

        # Train clean model
        model_clean = models[idx](num_features, num_labels, [600, 400, 200]).to('cuda')
        optimizer = torch.optim.Adam(model_clean.parameters(), lr=1e-4, weight_decay=1e-4)

        for epoch in range(501):
            loss = train(model_clean, train_data, optimizer)
            if epoch % 100 == 0:
                print(f'Model: Epoch {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {test(model_clean, train_data):.4f}, '
                      f'Val: {test(model_clean, val_data):.4f}, '
                      f'Test: {test(model_clean, test_data):.4f}')

        # Prepare backdoor data
        model_backdoor = models[idx](num_features, num_labels, [600, 400, 200]).to('cuda')
        poison_mask = torch.bernoulli(torch.full((train_data.num_features,), r, device='cuda')).bool()

        poison_nodes = (train_data.y != poison_label)
        poison_nodes &= torch.bernoulli(torch.full((len(train_data.x),), r, device='cuda')).bool()

        poison_data = copy.deepcopy(train_data)
        poison_data.x[poison_nodes.unsqueeze(1) & poison_mask.unsqueeze(0)] = pattern
        poison_data.y[poison_nodes] = poison_label

        optimizer = torch.optim.Adam(model_backdoor.parameters(), lr=1e-4, weight_decay=1e-4)
        start_time = time.time()

        for epoch in range(501):
            loss = train(model_backdoor, poison_data, optimizer)
            if epoch % 100 == 0:
                print(f'Model: Epoch {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {test(model_backdoor, train_data):.4f}, '
                      f'Val: {test(model_backdoor, val_data):.4f}, '
                      f'Test: {test(model_backdoor, test_data):.4f}')

        end_time = time.time()

        # Evaluate fidelity
        with open(filename, 'a') as file:
            file.write(f'{test(model_backdoor, train_data):.4f}, '
                       f'{test(model_backdoor, val_data):.4f}, '
                       f'{test(model_backdoor, test_data):.4f}, '
                       f'{end_time - start_time:.4f},')

        # Evaluate effectiveness
        poison_data = copy.deepcopy(test_data)
        poison_nodes = (test_data.y != poison_label)
        poison_nodes &= torch.bernoulli(torch.full((len(test_data.x),), r, device='cuda')).bool()
        poison_data.x[poison_nodes.unsqueeze(1) & poison_mask.unsqueeze(0)] = pattern
        bar_clean = (model_clean(poison_data.x, poison_data.edge_index).argmax(dim=1)[poison_nodes] == poison_label).float().mean().item()
        bar_backdoor = (model_backdoor(poison_data.x, poison_data.edge_index).argmax(dim=1)[poison_nodes] == poison_label).float().mean().item()

        with open(filename, 'a') as file:
            file.write(f'{bar_clean:.4f},{bar_backdoor:.4f},')

        # Evaluate misuse and use backdoor
        with open(filename, 'a') as file:
            model_backdoor.eval()
            y = model_backdoor(test_data.x, test_data.edge_index).argmax(dim=1)
            misuse = (y[poison_nodes] == poison_label).sum() / poison_nodes.sum().item()
            y = model_backdoor(poison_data.x, poison_data.edge_index).argmax(dim=1)
            use = (y[poison_nodes] == poison_label).sum() / poison_nodes.sum().item()
            file.write(f'{misuse:.4f},')
            file.write(f'{use:.4f},')

        # Model extraction
        sub_nodes = torch.zeros(train_data.num_nodes, dtype=torch.bool, device='cuda')
        sub_nodes[torch.randperm(train_data.num_nodes)[:train_data.num_nodes * 4 // 5]] = True
        sub_train = Data(x=train_data.x[sub_nodes],
                         edge_index=subgraph(sub_nodes, train_data.edge_index, relabel_nodes=True)[0],
                         y=train_data.y[sub_nodes])

        soft_output = model_backdoor(sub_train.x, sub_train.edge_index).softmax(dim=1).detach()

        for ExtractModel in [SAGE, SSG]:
            model_extracted = ExtractModel(sub_train.x.shape[1], soft_output.shape[1], [600, 400, 200]).to('cuda')
            model_extracted = model_extract(model_extracted, sub_train, soft_output)
            acc = test(model_extracted, test_data)
            bar = (model_extracted(poison_data.x, poison_data.edge_index).argmax(dim=1)[poison_nodes] == poison_label).float().mean().item()
            with open(filename, 'a') as file:
                file.write(f'{acc:.4f},{bar:.4f},')

        # Robustness: pruning
        tac_list, bar_list = [], []
        for rate in range(11):
            model_p = model_pruning(copy.deepcopy(model_backdoor), rate / 10)
            tac_list.append(test(model_p, test_data) * 100)
            yt = model_p(poison_data.x, poison_data.edge_index).argmax(dim=1)
            bar = (yt[poison_nodes] == poison_label).float().mean().item()
            bar_list.append(bar)

        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file, lineterminator='')
            writer.writerow(np.round(tac_list, 2).tolist() + np.round(bar_list, 3).tolist())
            file.write(",")

        # Robustness: fine-tuning
        model_ft = copy.deepcopy(model_backdoor)
        optimizer = torch.optim.Adam(model_ft.parameters(), lr=args.lr / 2)
        tac_list, bar_list = [], []

        for epoch in range(200):
            model_ft.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model_ft(val_data.x, val_data.edge_index), val_data.y)
            loss.backward()
            optimizer.step()

            tac_list.append(test(model_ft, test_data) * 100)
            yt = model_ft(poison_data.x, poison_data.edge_index).argmax(dim=1)
            bar = (yt[poison_nodes] == poison_label).float().mean().item()
            bar_list.append(bar)

            if epoch % 10 == 0:
                print("loss:", loss.item())

        with open(filename, 'a', newline='') as file:
            csv.writer(file).writerow(np.round(tac_list, 2).tolist() + np.round(bar_list, 3).tolist())
            file.write("\n")

        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    WGB(args=args)
