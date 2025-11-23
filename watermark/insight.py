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
    data = load_data(args)
    model_o = torch.load(args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm, weights_only=False)
    model_o.eval()
    # embedding watermark
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'insight3/'
    model_w, wm, wmk, trigger, _ = setting(model_o, None, data, args)
    model_w.eval()

    if args.paradigm == 'transductive':
        y_o = model_o(data.x, data.edge_index).softmax(dim=1)[data.test_mask]
        y_o = y_o.detach().cpu().numpy()
        ari_o = adjusted_rand_score(np.argmax(y_o, axis=1), data.y[data.test_mask].cpu().numpy())

        y_w = model_w(data.x, data.edge_index).softmax(dim=1)[data.test_mask]
        y_w = y_w.detach().cpu().numpy()
        ari_w = adjusted_rand_score(np.argmax(y_w, axis=1), data.y[data.test_mask].cpu().numpy())

        tsne_results_o = tsne.fit_transform(y_o)
        tsne_results_w = tsne.fit_transform(y_w)
        tsne_results = np.hstack((data.y[data.test_mask].detach().cpu().numpy().reshape(-1, 1), tsne_results_o, tsne_results_w))
        df = pd.DataFrame(tsne_results, columns=['label', 'o_dim1', 'o_dim2', 'w_dim1', 'w_dim2'])
        df['o_ARI'] = ari_o
        df['w_ARI'] = ari_w

        filename = args.results_path + args.dataset + '_' + args.paradigm + '_setting' + str(args.setting) + '_tsne.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)

    elif args.paradigm == 'inductive':
        y_o = model_o(data[2].x, data[2].edge_index).softmax(dim=1)
        y_o = y_o.detach().cpu().numpy()
        ari_o = adjusted_rand_score(np.argmax(y_o, axis=1), data[2].y.cpu().numpy())

        y_w = model_w(data[2].x, data[2].edge_index).softmax(dim=1)
        y_w = y_w.detach().cpu().numpy()
        ari_w = adjusted_rand_score(np.argmax(y_w, axis=1), data[2].y.cpu().numpy())

        tsne_results_o = tsne.fit_transform(y_o)
        tsne_results_w = tsne.fit_transform(y_w)
        tsne_results = np.hstack((data[2].y.detach().cpu().numpy().reshape(-1, 1), tsne_results_o, tsne_results_w))
        df = pd.DataFrame(tsne_results, columns=['label', 'o_dim1', 'o_dim2', 'w_dim1', 'w_dim2'])
        df['o_ARI'] = ari_o
        df['w_ARI'] = ari_w

        filename = args.results_path + args.dataset + '_' + args.paradigm + '_setting' + str(args.setting) + '_tsne.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)

    else:
        raise ValueError('Error: Wrong paradigm!')

    args.results_path = results_path


def insight2(args):
    data = load_data(args)
    model_o = torch.load(args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm, weights_only=False)
    model_o.eval()
    # embedding watermark
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'insight2/'
    model_w, wm, wmk, trigger, _ = setting(model_o, None, data, args)
    model_w.eval()
    wm = wm.detach().cpu().numpy()

    y_o = model_o(trigger.x, trigger.edge_index).softmax(dim=1)
    v_o = LDDE(y_o, trigger.x, trigger.edge_index[:, wmk]).flatten()
    hms_o = 1 - hamming_loss(wm, (v_o > 0).int().detach().cpu().numpy())

    y_w = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
    v_w = LDDE(y_w, trigger.x, trigger.edge_index[:, wmk]).flatten()
    hms_w = 1 - hamming_loss(wm, (v_w > 0).int().detach().cpu().numpy())

    v = np.vstack((wm, v_o.detach().cpu().numpy(), v_w.detach().cpu().numpy())).T
    df = pd.DataFrame(v, columns=['WM', 'LDDE_o', 'LDDE_w'])
    df['HMS_o'] = hms_o
    df['HMS_w'] = hms_w

    filename = args.results_path + args.dataset + '_' + args.paradigm + '_setting' + str(args.setting) + '_ldde.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

    args.results_path = results_path


def watermark_collision(args):
    data = load_data(args)
    model_o = torch.load(args.model_path + args.dataset + '/' + args.model, weights_only=False)
    model_o.eval()
    # generate trigger
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'collision/'
    _, _, wmk, trigger, _ = setting(model_o, None, data, args)

    watermarks = []
    ldde_list = []
    for i in range(args.model_num):
        wm = watermark_string_generation(args)
        if args.setting == 1 :
            model_w = watermark_embedding_1(copy.deepcopy(model_o), data, wm, wmk, trigger, args)
        elif args.setting == 2 :
            model_w = watermark_embedding_2(copy.deepcopy(model_o), data, wm, wmk, trigger, args)
        else:
            raise ValueError('Error: Wrong setting!')
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
    filename = args.results_path + 'collision/' + args.paradigm + '/setting' + str(args.setting) + '/' + args.dataset + '_collision.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)
            csv.writer(file).writerows(new_line.reshape(-1, 1))



def multibit(args):
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'multibit/'
    filename = args.results_path + args.paradigm + '/setting' + str(args.setting) + '/' + args.dataset + '_multibit.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    headers = ['Nw', 'Mi HMS', 'Mw HMS', 'Test CE','Test BCE','Pruning CE','Pruning BCE','Fine-tuning CE',' Fine-tuning BCE']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    data = load_data(args)
    model_name = args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm
    args.random_seed = torch.manual_seed(int(time.time() * 100))
    if os.path.exists(model_name):
        model = torch.load(model_name, weights_only=False)
    else:
        raise ValueError('Error: Model not found!')
    n_wm_default = copy.deepcopy(args.n_wm)
    n_wm = [16, 32, 64, 128, 256]
    for nwm in n_wm:
        if nwm == 256 and args.dataset == 'Cora':
            continue
        args.n_wm = nwm
        aline = [args.n_wm]

        model_w, wm, wmk, trigger, model_i = setting(copy.deepcopy(model), copy.deepcopy(model), data, args)
        model_w.eval()
        wm = wm.to(torch.float32)
        if args.paradigm == 'inductive':
            y_hat = model_w(data[2].x, data[2].edge_index)
            #y_hat = y_hat.detach().cpu().numpy()
            #ari = adjusted_rand_score(np.argmax(y_hat, axis=1), data[2].y.cpu().numpy())
            ce = F.cross_entropy(y_hat, data[2].y)
            y_pred = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
            v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
            bce = F.binary_cross_entropy_with_logits(v, wm)
            hms_i = watermark_verification(model_i, wm, wmk, trigger)
            hms_w = watermark_verification(model_w, wm, wmk, trigger)
            aline.append(hms_i)
            aline.append(hms_w)
            aline.append(ce.detach().cpu().numpy())
            aline.append(bce.detach().cpu().numpy())

            #robust
            model_w2 = model_pruning(copy.deepcopy(model_w), 0.7)
            y_hat = model_w2(data[2].x, data[2].edge_index)
            #ari = adjusted_rand_score(np.argmax(y_hat, axis=1), data[2].y.cpu().numpy())
            ce = F.cross_entropy(y_hat, data[2].y)
            y_pred = model_w2(trigger.x, trigger.edge_index).softmax(dim=1)
            v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
            #hms = 1 - hamming_loss(wm.detach().cpu().numpy(), (v > 0).int().detach().cpu().numpy())
            bce = F.binary_cross_entropy_with_logits(v, wm)
            aline.append(ce.detach().cpu().numpy())
            aline.append(bce.detach().cpu().numpy())

            model_w2, _, _ = fine_tuning(copy.deepcopy(model_w), data, wm, wmk, trigger, args)
            y_hat = model_w2(data[2].x, data[2].edge_index)
            ce = F.cross_entropy(y_hat, data[2].y)
            #ari = adjusted_rand_score(np.argmax(y_hat, axis=1), data[2].y.cpu().numpy())
            y_pred = model_w2(trigger.x, trigger.edge_index).softmax(dim=1)
            v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
            bce = F.binary_cross_entropy_with_logits(v, wm)
            #hms = 1 - hamming_loss(wm.detach().cpu().numpy(), (v > 0).int().detach().cpu().numpy())
            aline.append(ce.detach().cpu().numpy())
            aline.append(bce.detach().cpu().numpy())
        elif args.paradigm == 'transductive':
            y_hat = model_w(data.x, data.edge_index)
            ce = F.cross_entropy(y_hat[data.test_mask], data.y[data.test_mask])
            #ari = adjusted_rand_score(np.argmax(y_hat, axis=1), data.y[data.test_mask].cpu().numpy())
            y_pred = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
            v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
            #hms = 1 - hamming_loss(wm.detach().cpu().numpy(), (v > 0).int().detach().cpu().numpy())
            bce = F.binary_cross_entropy_with_logits(v, wm)
            hms_i = watermark_verification(model_i, wm, wmk, trigger)
            hms_w = watermark_verification(model_w, wm, wmk, trigger)
            aline.append(hms_i)
            aline.append(hms_w)
            aline.append(ce.detach().cpu().numpy())
            aline.append(bce.detach().cpu().numpy())

            # robust
            model_w2 = model_pruning(copy.deepcopy(model_w), 0.7)
            y_hat = model_w2(data.x, data.edge_index)
            ce = F.cross_entropy(y_hat[data.test_mask], data.y[data.test_mask])
            #ari = adjusted_rand_score(np.argmax(y_hat.softmax(dim=1).detach().cpu().numpy(), axis=1), data[2].y.cpu().numpy())
            y_pred = model_w2(trigger.x, trigger.edge_index).softmax(dim=1)
            v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
            #hms = 1 - hamming_loss(wm.detach().cpu().numpy(), (v > 0).int().detach().cpu().numpy())
            bce = F.binary_cross_entropy_with_logits(v, wm)
            aline.append(ce.detach().cpu().numpy())
            aline.append(bce.detach().cpu().numpy())

            model_w2, _, _ = fine_tuning(copy.deepcopy(model_w), data, wm, wmk, trigger, args)
            y_hat = model_w2(data.x, data.edge_index)
            ce = F.cross_entropy(y_hat[data.test_mask], data.y[data.test_mask])
            #ari = adjusted_rand_score(np.argmax(y_hat.softmax(dim=1).detach().cpu().numpy(), axis=1), data[2].y.cpu().numpy())
            y_pred = model_w2(trigger.x, trigger.edge_index).softmax(dim=1)
            v = LDDE(y_pred, trigger.x, trigger.edge_index[:, wmk])
            #hms = 1 - hamming_loss(wm.detach().cpu().numpy(), (v > 0).int().detach().cpu().numpy())
            bce = F.binary_cross_entropy_with_logits(v, wm)
            aline.append(ce.detach().cpu().numpy())
            aline.append(bce.detach().cpu().numpy())

        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(aline)

    args.n_wm = n_wm_default
    args.results_path = results_path



def assess_insight(args):
    # insight2(args)
    # insight3(args)
    multibit(args)
    # watermark_collision(args)
