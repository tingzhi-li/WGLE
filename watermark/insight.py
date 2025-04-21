import copy

import numpy
import numpy as np
import pandas as pd
import csv

import torch
import torch_geometric
import networkx as nx
import os
from torch_geometric.datasets import Planetoid
from sklearn.metrics import hamming_loss, adjusted_rand_score
from sklearn.manifold import TSNE
from watermark.watermark import *
from utils.dataload import load_data
from watermark.assess import *
from watermark.robust import model_pruning, fine_tuning

def insight3(args):
    tsne = TSNE(n_components=2, random_state=42)
    train_data, val_data, test_data, num_features, num_labels = load_data(args)
    model_o = torch.load(args.model_path + args.dataset + '/' + args.model, weights_only=False)
    model = copy.deepcopy(model_o)
    model.eval()
    # embedding watermark
    model_w, wm, wmk, trigger, _ = setting(model, model, train_data, val_data, test_data, args)
    model_w.eval()

    y_o = model_o(train_data.x, train_data.edge_index).softmax(dim=1).detach().cpu().numpy()
    ari = adjusted_rand_score(np.argmax(y_o, axis=1), train_data.y.cpu().numpy())
    tsne_results_o = tsne.fit_transform(y_o)
    tsne_results_o = np.hstack((tsne_results_o, trigger.y.detach().cpu().numpy().reshape(-1, 1)))
    df = pd.DataFrame(tsne_results_o, columns=['Dimension 1', 'Dimension 2', 'Class'])
    df['ARI'] = ari
    filename = args.results_path + 'insight3/' + args.dataset + str(args.setting) + '_tsne_o.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

    y_w = model_w(trigger.x, trigger.edge_index).softmax(dim=1).detach().cpu().numpy()
    ari = adjusted_rand_score(np.argmax(y_w, axis=1), trigger.y.cpu().numpy())
    tsne_results_w = tsne.fit_transform(y_w)
    tsne_results_w = np.hstack((tsne_results_w, trigger.y.detach().cpu().numpy().reshape(-1, 1)))
    df = pd.DataFrame(tsne_results_w, columns=['Dimension 1', 'Dimension 2', 'Class'])
    df['ARI'] = ari
    df.to_csv(args.results_path + 'insight3/' + args.dataset + str(args.setting) + '_tsne_w.csv', index=False)


def insight2(args):
    train_data, val_data, test_data, num_features, num_labels = load_data(args)
    model_o = torch.load(args.model_path + args.dataset + '/' + args.model, weights_only=False)
    model = copy.deepcopy(model_o)
    model.eval()

    model_w, wm, wmk, trigger, _ = setting(model, model, train_data, val_data, test_data, args)
    y = model(trigger.x, trigger.edge_index).softmax(dim=1)
    v = LDDE(y, trigger.x, trigger.edge_index[:, wmk]).flatten()
    label = wm.detach().cpu().numpy()
    hamming_similarity = 1 - hamming_loss(label, (v > 0).int().detach().cpu().numpy())
    v = np.vstack((v.detach().cpu().numpy(), label)).T
    df = pd.DataFrame(v, columns=['LDDE', 'Label'])
    df['HMS'] = hamming_similarity
    filename = args.results_path + 'insight2/' + args.dataset + str(args.setting) + '_o.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

    model_w.eval()
    y = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
    v = LDDE(y, trigger.x, trigger.edge_index[:, wmk]).flatten()
    hamming_similarity = 1 - hamming_loss(label, (v > 0).int().detach().cpu().numpy())
    v = np.vstack((v.detach().cpu().numpy(), label)).T
    df = pd.DataFrame(v, columns=['LDDE', 'Label'])
    df['HMS'] = hamming_similarity

    filename = args.results_path + 'insight2/' + args.dataset + str(args.setting) + '_w.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)


def watermark_collision(args):
    train_data, val_data, test_data, num_features, num_labels = load_data(args)
    model_o = torch.load(args.model_path + args.dataset + '/' + args.model, weights_only=False)
    model = copy.deepcopy(model_o)
    model.eval()

    filename = args.results_path +  'collision/' + args.dataset+ '/setting' + str(args.setting) + '.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    model = copy.deepcopy(model_o)
    model.eval()

    # embedding watermark
    watermarks = []
    ldde_list = []
    if args.setting == 1:
        trigger = train_data
    else:
        edge_index = Planetoid(root=args.dataset_path, name='PubMed')[0].edge_index.detach().clone().to(device)
        trigger = trigger_generation(model_o, edge_index, args)

    for i in range(args.model_num):
        wm = watermark_string_generation(args)
        wmk = watermark_key_generation(model_o, trigger, wm, args)
        if args.setting == 1 or args.setting == 2:
            model_w = watermark_embedding_1(model_o, train_data, val_data, test_data, wm, wmk, trigger, args)
        else:
            model_w = watermark_embedding_2(model_o, train_data, val_data, test_data, wm, wmk, trigger, args)
        model_w.eval()
        y_hat = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
        v = LDDE(y_hat, trigger.x, trigger.edge_index[:, wmk])
        ldde_list.append(v.cpu().detach())
        watermarks.append(wm.cpu().detach())
        torch.cuda.empty_cache()
        print(f'No.{i} model_w')

    ldde_list = np.vstack(ldde_list)
    ldde_list = np.where(ldde_list < 0, 0, 1)
    watermarks = np.vstack(watermarks).astype(int)
    new_line = []
    for i in range(1, args.model_num):
        new_line.append((ldde_list == np.roll(watermarks, shift=i, axis=0)).sum(axis=1) / args.n_wm)
    new_line = np.array(new_line).flatten()
    headers = ['HMS']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)
            csv.writer(file).writerows(new_line.reshape(-1, 1))



def multibit(args):
    filename = args.results_path + 'multibit/' + args.dataset + '/setting' + str(args.setting) + '.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    headers = ['Nw','Test CE','Test BCE','Pruning CE','Pruning BCE','Fine-tuning CE',' Fine-tuning BCE']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    train_data, val_data, test_data, num_features, num_labels = load_data(args)
    model_name = args.model_path + args.dataset + '/' + args.model
    args.random_seed = torch.manual_seed(int(time.time() * 100))
    if os.path.exists(model_name):
        model = torch.load(model_name, weights_only=False)
    n_wm_default = copy.deepcopy(args.n_wm)
    n_wm = [20, 100, 200, 500]
    for nwm in n_wm:
        args.n_wm = nwm
        if args.dataset == 'Cora' and args.n_wm == 500:
            continue
        aline = [args.n_wm]

        setting(copy.deepcopy(model), copy.deepcopy(model), train_data, val_data, test_data, args)
        model_w, wm, wmk, trigger, _ = setting(copy.deepcopy(model), copy.deepcopy(model), train_data, val_data, test_data,
                                                         args)
        model_w.eval()
        y_hat = model_w(test_data.x, test_data.edge_index)
        CE = F.cross_entropy(y_hat, test_data.y)
        y_pred = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
        v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
        BCE = F.binary_cross_entropy_with_logits(v.flatten(), wm.to(torch.float32))
        aline.append(CE.item())
        aline.append(BCE.item())

        model_w2 = model_w
        # robust
        model_w = model_pruning(copy.deepcopy(model_w2), 0.7)
        y_hat = model_w(test_data.x, test_data.edge_index)
        CE = F.cross_entropy(y_hat, test_data.y)
        y_pred = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
        v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
        BCE = F.binary_cross_entropy_with_logits(v.flatten(), wm.to(torch.float32))
        aline.append(CE.item())
        aline.append(BCE.item())

        model_w, _, _ = fine_tuning(copy.deepcopy(model_w2), val_data, test_data, wm, wmk, trigger)
        y_hat = model_w(test_data.x, test_data.edge_index)
        CE = F.cross_entropy(y_hat, test_data.y)
        y_pred = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
        v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
        BCE = F.binary_cross_entropy_with_logits(v.flatten(), wm.to(torch.float32))
        aline.append(CE.item())
        aline.append(BCE.item())

        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(aline)

    args.n_wm = n_wm_default




def assess_insight(args):
    insight2(args)
    insight3(args)
    multibit(args)
    watermark_collision(args)


