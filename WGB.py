from utils.dataload import *
from watermark.robust import *
from utils.utils import *

import os
import csv
import time
import copy
import torch
import random
import numpy as np
import torch.nn.functional as F

from utils.dataload import load_data
from watermark.robust import model_pruning, model_extract
from utils.utils import test, train
from utils.config import parse_args
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from utils.models import *


def WGB(r=0.10, poison_label=2, args=None):
    datasets = ['Cora', 'DBLP', 'CS', 'Physics', 'Blog', 'Photo']
    models = ['GAT', 'GTF', 'SSG', 'GCNv2', 'ARMA', 'SAGE']
    headers = [
        'Train Acc', 'Test Acc', 'Time', 'Mc BAR', 'Mb BAR',
    ]
    headers += ['MEA test_acc', 'MEA Mb bar',]
    headers += ['70% Mc test_acc', '70% Mc bar', '70% Mb test_acc', '70% Mb bar',]
    headers += ['200 Mc test_acc', '200 Mc bar', '200 Mb test_acc', '200 Mb bar']

    args.paradigm = 'inductive'
    for i in range(6):
        args.dataset = datasets[i]
        args.model = models[i]
        filename = 'WGB/'+ args.dataset + "_ind.csv"

        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, mode='w', newline='') as file:
                csv.writer(file).writerow(headers)

        aline = []
        data = load_data(args)
        num_features = data[0].num_features
        num_classes = data[0].y.max().item() + 1

        # Train clean model
        model_clean = load_model(num_features, num_classes, args)
        optimizer = torch.optim.Adam(model_clean.parameters(), lr=1e-4, weight_decay=1e-4)
        for epoch in range(501):
            loss = train(model_clean, data, optimizer, args)
            if epoch % 100 == 0:
                train_acc, test_acc = test(model_clean, data, args)
                print(
                    f'Clean model is training... Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

        # Prepare backdoor data
        pattern = random.random()
        model_backdoor = load_model(num_features, num_classes, args)
        n_col = int(len(data[0].x[0]) * 0.2)

        poison_nodes = (data[0].y != poison_label)
        poison_nodes &= torch.bernoulli(torch.full((len(data[0].x),), r, device='cuda')).bool()

        poison_data = copy.deepcopy(data)
        poison_data[0].x[poison_nodes, :n_col] = pattern
        poison_data[0].y[poison_nodes] = poison_label

        optimizer = torch.optim.Adam(model_backdoor.parameters(), lr=1e-4, weight_decay=1e-4)
        start_time = time.time()

        for epoch in range(501):
            loss = train(model_backdoor, data, optimizer, args)
            loss = train(model_backdoor, poison_data, optimizer, args)
            if epoch % 100 == 0:
                train_acc, test_acc = test(model_backdoor, data, args)
                print(
                    f'Backdoor model is training... Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

        end_time = time.time()

        # Evaluate fidelity
        train_acc, test_acc = test(model_backdoor, data, args)
        aline.append(train_acc)
        aline.append(test_acc)
        aline.append(end_time - start_time)

        # Evaluate effectiveness
        poison_data = poison_data[0]
        bar_clean = (model_clean(poison_data.x, poison_data.edge_index)[poison_nodes].argmax(dim=1) == poison_label).sum() / sum(poison_nodes)
        bar_backdoor = (model_backdoor(poison_data.x, poison_data.edge_index)[poison_nodes].argmax(dim=1) == poison_label).sum() / sum(poison_nodes)

        aline.append(bar_clean.item())
        aline.append(bar_backdoor.item())

        # Evaluate misuse and use backdoor
        # with open(filename, 'a') as file:
        #     model_backdoor.eval()
        #     y = model_backdoor(test_data.x, test_data.edge_index).argmax(dim=1)
        #     misuse = (y[poison_nodes] == poison_label).sum() / poison_nodes.sum().item()
        #     y = model_backdoor(poison_data.x, poison_data.edge_index).argmax(dim=1)
        #     use = (y[poison_nodes] == poison_label).sum() / poison_nodes.sum().item()
        #     file.write(f'{misuse:.4f},')
        #     file.write(f'{use:.4f},')

        #Model extraction
        soft_output = model_backdoor(data[1].x, data[1].edge_index).softmax(dim=1).detach()

        for ExtractModel in [SAGE]:
            model_extracted = ExtractModel(num_features, num_classes, args.hidden_channels).to('cuda')
            model_extracted = model_extract(model_extracted, data, soft_output, args, False)
            train_acc, test_acc = test(model_extracted, data, args)
            bar = (model_extracted(poison_data.x, poison_data.edge_index)[poison_nodes].argmax(dim=1) == poison_label).sum() / sum(poison_nodes)
            aline.append(test_acc)
            aline.append(bar.item())


        # Robustness: pruning
        for rate in range(7, 8):
            model_p = model_pruning(copy.deepcopy(model_clean), rate / 10)
            aline.append(test(model_p, data, args)[1])
            yt = model_p(poison_data.x, poison_data.edge_index).argmax(dim=1)
            bar = (yt[poison_nodes] == poison_label).sum() / sum(poison_nodes)
            aline.append(bar.item())

            model_p = model_pruning(copy.deepcopy(model_backdoor), rate / 10)
            aline.append(test(model_p, data, args)[1])
            yt = model_p(poison_data.x, poison_data.edge_index).argmax(dim=1)
            bar = (yt[poison_nodes] == poison_label).sum() / sum(poison_nodes)
            aline.append(bar.item())

        # Robustness: fine-tuning
        model_ft = copy.deepcopy(model_clean)
        optimizer = torch.optim.Adam(model_ft.parameters(), lr=args.lr / 2)

        for epoch in range(200):
            model_ft.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model_ft(data[1].x, data[1].edge_index), data[1].y)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print("fine-tuning loss:", loss.item())
        aline.append(test(model_ft, data, args)[1])
        yt = model_ft(poison_data.x, poison_data.edge_index).argmax(dim=1)
        bar = (yt[poison_nodes] == poison_label).sum() / sum(poison_nodes)
        aline.append(bar.item())


        model_ft = copy.deepcopy(model_backdoor)
        optimizer = torch.optim.Adam(model_ft.parameters(), lr=args.lr / 2)

        for epoch in range(200):
            model_ft.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model_ft(data[1].x, data[1].edge_index), data[1].y)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print("fine-tuning loss:", loss.item())
        aline.append(test(model_ft, data, args)[1])
        yt = model_ft(poison_data.x, poison_data.edge_index).argmax(dim=1)
        bar = (yt[poison_nodes] == poison_label).sum() / sum(poison_nodes)
        aline.append(bar.item())

        with open(filename, 'a') as file:
            csv.writer(file).writerow(aline)

        torch.cuda.empty_cache()

    args.paradigm = 'transductive'
    for i in range(6):
        args.dataset = datasets[i]
        args.model = models[i]
        filename = 'WGB/' + args.dataset + "_tra.csv"

        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, mode='w', newline='') as file:
                csv.writer(file).writerow(headers)

        aline = []
        data = load_data(args)
        num_features = data.num_features
        num_classes = data.y.max().item() + 1

        # Train clean model
        model_clean = load_model(num_features, num_classes, args)
        optimizer = torch.optim.Adam(model_clean.parameters(), lr=1e-4, weight_decay=1e-4)
        for epoch in range(501):
            loss = train(model_clean, data, optimizer, args)
            if epoch % 100 == 0:
                train_acc, test_acc = test(model_clean, data, args)
                print(
                    f'Clean model is training... Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

        # Prepare backdoor data
        pattern = random.random()
        model_backdoor = load_model(num_features, num_classes, args)
        n_col = int(len(data.x[0]) * 0.2)

        poison_nodes = (data.y != poison_label) & data.train_mask
        poison_nodes &= torch.bernoulli(torch.full((len(data.x),), r, device='cuda')).bool()

        poison_data = copy.deepcopy(data)
        poison_data.x[poison_nodes, :n_col] = pattern
        poison_data.y[poison_nodes] = poison_label

        optimizer = torch.optim.Adam(model_backdoor.parameters(), lr=1e-4, weight_decay=1e-4)
        start_time = time.time()

        for epoch in range(501):
            loss = train(model_backdoor, data, optimizer, args)
            loss = train(model_backdoor, poison_data, optimizer, args)
            if epoch % 100 == 0:
                train_acc, test_acc = test(model_backdoor, data, args)
                print(
                    f'Backdoor model is training... Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

        end_time = time.time()

        # Evaluate fidelity
        train_acc, test_acc = test(model_backdoor, data, args)
        aline.append(train_acc)
        aline.append(test_acc)
        aline.append(end_time - start_time)

        # Evaluate effectiveness
        bar_clean = (model_clean(poison_data.x, poison_data.edge_index)[
                         poison_nodes].argmax(dim=1) == poison_label).sum() / sum(poison_nodes)
        bar_backdoor = (model_backdoor(poison_data.x, poison_data.edge_index)[
                            poison_nodes].argmax(dim=1) == poison_label).sum() / sum(poison_nodes)

        aline.append(bar_clean.item())
        aline.append(bar_backdoor.item())

        # Evaluate misuse and use backdoor
        # with open(filename, 'a') as file:
        #     model_backdoor.eval()
        #     y = model_backdoor(test_data.x, test_data.edge_index).argmax(dim=1)
        #     misuse = (y[poison_nodes] == poison_label).sum() / poison_nodes.sum().item()
        #     y = model_backdoor(poison_data.x, poison_data.edge_index).argmax(dim=1)
        #     use = (y[poison_nodes] == poison_label).sum() / poison_nodes.sum().item()
        #     file.write(f'{misuse:.4f},')
        #     file.write(f'{use:.4f},')

        # Model extraction
        soft_output = model_backdoor(data.x, data.edge_index).softmax(dim=1).detach()[data.val_mask]

        for ExtractModel in [SAGE]:
            model_extracted = ExtractModel(num_features, num_classes, args.hidden_channels).to('cuda')
            model_extracted = model_extract(model_extracted, data, soft_output, args, False)
            train_acc, test_acc = test(model_extracted, data, args)
            bar = (model_extracted(poison_data.x, poison_data.edge_index)[poison_nodes].argmax(dim=1) == poison_label).sum() / sum(poison_nodes)
            aline.append(test_acc)
            aline.append(bar.item())

        # Robustness: pruning
        for rate in range(7, 8):
            model_p = model_pruning(copy.deepcopy(model_clean), rate / 10)
            aline.append(test(model_p, data, args)[1])
            yt = model_p(poison_data.x, poison_data.edge_index).argmax(dim=1)
            bar = (yt[poison_nodes] == poison_label).sum() / sum(poison_nodes)
            aline.append(bar.item())

            model_p = model_pruning(copy.deepcopy(model_backdoor), rate / 10)
            aline.append(test(model_p, data, args)[1])
            yt = model_p(poison_data.x, poison_data.edge_index).argmax(dim=1)
            bar = (yt[poison_nodes] == poison_label).sum() / sum(poison_nodes)
            aline.append(bar.item())

        # Robustness: fine-tuning
        model_ft = copy.deepcopy(model_clean)
        optimizer = torch.optim.Adam(model_ft.parameters(), lr=args.lr / 2)

        for epoch in range(200):
            model_ft.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model_ft(data.x, data.edge_index)[data.val_mask], data.y[data.val_mask])
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print("fine-tuning loss:", loss.item())
        aline.append(test(model_ft, data, args)[1])
        yt = model_ft(poison_data.x, poison_data.edge_index).argmax(dim=1)
        bar = (yt[poison_nodes] == poison_label).sum() / sum(poison_nodes)
        aline.append(bar.item())

        model_ft = copy.deepcopy(model_backdoor)
        optimizer = torch.optim.Adam(model_ft.parameters(), lr=args.lr / 2)

        for epoch in range(200):
            model_ft.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model_ft(data.x, data.edge_index)[data.val_mask], data.y[data.val_mask])
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print("fine-tuning loss:", loss.item())
        aline.append(test(model_ft, data, args)[1])
        yt = model_ft(poison_data.x, poison_data.edge_index).argmax(dim=1)
        bar = (yt[poison_nodes] == poison_label).sum() / sum(poison_nodes)
        aline.append(bar.item())

        with open(filename, 'a') as file:
            csv.writer(file).writerow(aline)

        torch.cuda.empty_cache()




if __name__ == '__main__':
    args = parse_args()
    for i in range(5):
        WGB(args=args)
