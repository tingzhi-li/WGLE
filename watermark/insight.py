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
    model_o = torch.load(args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm, weights_only=False)
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
    args.results_path = results_path



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



def hard_label(args):
    def LDDE_hardlabel(y_logits, x, edge):
        """
        Hard-label LDDE: converts model logits to one-hot vectors via argmax,
        then computes cosine-similarity-based distance difference on edges.
        Uses a straight-through estimator so gradients flow through softmax
        during training while the forward pass uses strict one-hot decisions.
        """
        y_soft = torch.softmax(y_logits, dim=1)
        y_pred = y_logits.argmax(dim=1)
        y_hard = F.one_hot(y_pred, num_classes=y_logits.size(1)).float()
        # Straight-through estimator: hard one-hot in forward, softmax gradient in backward
        y_hardlabel = y_hard + y_soft - y_soft.detach()
        distance_x = F.cosine_similarity(x[edge[0]], x[edge[1]])
        distance_y = F.cosine_similarity(y_hardlabel[edge[0]], y_hardlabel[edge[1]])
        return distance_y - distance_x

    def hardlabel_watermark_verification(model, wm, wk, trigger):
        """Extract watermark bits using LDDE_hardlabel and compute Hamming similarity."""
        model.eval()
        trigger_logits = model(trigger.x, trigger.edge_index)
        v = LDDE_hardlabel(trigger_logits, trigger.x, trigger.edge_index[:, wk]).flatten()
        wme = torch.where(v < 0, 0, 1)
        hms = int((wme == wm).sum()) / len(wm)
        return hms

    def trigger_generation_hardlabel(model_o, edge_index, args):
        model_o.eval()
    
        num_nodes = edge_index.max() + 1
        if isinstance(model_o, GCNv2):
            num_feat = model_o.fc.in_channels
        elif isinstance(model_o, GIN):
            num_feat = model_o.layers[0].nn.in_channels
        else:
            num_feat = model_o.layers[0].in_channels
        x = F.hardtanh(torch.randn((num_nodes, num_feat))).to(args.device)
        x.requires_grad_(True)
        loss_copy = 1000
        x_copy = None
        optimizer_data = torch.optim.Adam([x], lr=args.trigger_lr)
        mask = edge_index[0] < edge_index[1]
    
        for epoch in range(args.trigger_epochs):
            y_hat = model_o(F.hardtanh(x), edge_index).softmax(dim=1)
            v = LDDE_hardlabel(y_hat, F.hardtanh(x), edge_index[:, mask])
            optimizer_data.zero_grad()
            loss1 = torch.mean(torch.abs(v))
            loss2 = torch.mean(1 / (1 - torch.abs(F.cosine_similarity(F.hardtanh(x)[edge_index[0, mask]], F.hardtanh(x)[edge_index[1, mask]]))))
            loss = loss1 + 1e-4 * loss2
            loss.backward()
            optimizer_data.step()
    
            if loss1 < loss_copy:
                loss_copy = loss1.detach().clone().item()
                x_copy = F.hardtanh(x).detach().clone()
            if epoch % 50 == 0:
                print(f'Trigger is generating. Epoch:{epoch}, Loss1:{loss1:.4f}, Loss2:{loss2:.4f}')
    
        x = x_copy
        torch.cuda.empty_cache()
        node2vec = torch_geometric.nn.Node2Vec(edge_index, embedding_dim=128, walk_length=20, context_size=10,
                                                   walks_per_node=10, num_negative_samples=1, p=1.0, q=1.0,
                                                   sparse=True, ).to(args.device)
        loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=args.trigger_lr)
        for epoch in range(args.trigger_epochs//10):
            node2vec.train()
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw.to(args.device), neg_rw.to(args.device))
                loss.backward()
                optimizer.step()
            if epoch % 5 == 0:
                print(f'Node2Vec is executing. Epoch:{epoch}, Loss:{loss:.4f}')
        node2vec.eval()
        node_embeddings = node2vec.embedding.weight.data
        edge_embeddings = torch.abs(node_embeddings[edge_index[0]] - node_embeddings[edge_index[1]])
        edge_embeddings = edge_embeddings.detach().cpu().numpy()
        dbscan = DBSCAN(eps=1.5, min_samples=10)
        edge_labels = dbscan.fit_predict(edge_embeddings)
        edge_attr = torch.from_numpy(edge_labels == -1).to(args.device)
        
        trigger = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(args.device)
        print(trigger)
        return trigger

    # ---------- Load data and original model ----------
    data = load_data(args)
    model_o = torch.load(args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm, weights_only=False)
    model_o.eval()

    # ---------- Set up results path ----------
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'hard_label/'

    # ---------- Prepare trigger and generate watermark / key ----------
    if args.setting == 1:
        if args.paradigm == 'transductive':
            trigger = data
        elif args.paradigm == 'inductive':
            trigger = data[0]
        else:
            raise ValueError('Error: Wrong paradigm!')
    elif args.setting == 2:
        edge_index = CitationFull(root=args.dataset_path, name='CiteSeer')[0].edge_index.detach().clone().to(args.device)
        trigger = trigger_generation_hardlabel(model_o, edge_index, args)
    else:
        raise ValueError('Error: Wrong setting!')

    wm = watermark_string_generation(args)
    wmk = watermark_key_generation(model_o, trigger, args)
    wm_float = wm.to(torch.float32)

    # ---------- Watermark Embedding using LDDE_hardlabel ----------
    model_w = copy.deepcopy(model_o)
    optimizer_model = torch.optim.Adam(model_w.parameters(), lr=args.wm_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=128, eta_min=1e-5)

    for epoch in range(args.max_epochs):
        model_w.train()
        optimizer_model.zero_grad()

        trigger_logits = model_w(trigger.x, trigger.edge_index)

        # Task fidelity loss
        if args.paradigm == 'transductive':
            y = model_w(data.x, data.edge_index)
            loss1 = F.cross_entropy(y[data.train_mask], data.y[data.train_mask])
        elif args.paradigm == 'inductive':
            y = model_w(data[0].x, data[0].edge_index)
            loss1 = F.cross_entropy(y, data[0].y)
        else:
            raise ValueError('Error: Wrong paradigm!')

        # Watermark loss: BCE between LDDE_hardlabel values and watermark bits
        v = LDDE_hardlabel(trigger_logits, trigger.x, trigger.edge_index[:, wmk])
        loss2 = F.binary_cross_entropy_with_logits(v.flatten(), wm_float)
        loss = loss1 + args.coe * loss2

        loss.backward()
        optimizer_model.step()
        scheduler.step()

        if epoch % 20 == 0:
            train_acc, test_acc = test(model_w, data, args)
            bcr = hardlabel_watermark_verification(model_w, wm, wmk, trigger)
            print(f'hard_label embedding. Epoch:{epoch}, Loss1:{loss1:.4f}, Loss2:{loss2:.4f}, '
                  f'Train_acc:{train_acc:.4f}, Test_acc:{test_acc:.4f}, HMS:{bcr}')
            if (bcr > 0.99) and (epoch >= 100):
                break

    model_w.eval()
    _, test_acc = test(model_w, data, args)
    # ---------- Watermark Extraction & Comparison ----------
    hms_w = hardlabel_watermark_verification(model_w, wm, wmk, trigger)

    y_o_logits = model_o(trigger.x, trigger.edge_index)
    v_o = LDDE_hardlabel(y_o_logits, trigger.x, trigger.edge_index[:, wmk]).flatten()
    hms_o = 1 - hamming_loss(wm.detach().cpu().numpy(), (v_o > 0).int().detach().cpu().numpy())

    y_w_logits = model_w(trigger.x, trigger.edge_index)
    v_w = LDDE_hardlabel(y_w_logits, trigger.x, trigger.edge_index[:, wmk]).flatten()

    # Build and save results DataFrame
    #v = np.vstack((wm.detach().cpu().numpy(), v_o.detach().cpu().numpy(), v_w.detach().cpu().numpy())).T
    #df = pd.DataFrame(v, columns=[])
    df = pd.DataFrame({'TAC': [test_acc], 'HMS_o': [hms_o], 'HMS_w': [hms_w]})
    
    filename = args.results_path + args.dataset + '_' + args.paradigm + '_setting' + str(args.setting) + '_hardlabel.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)
    
    args.results_path = results_path



def ablation_ldde(args):
    # ═══════════════════════════════════════════════════════════════════════
    # LDDE variant definitions for ablation study
    # ═══════════════════════════════════════════════════════════════════════

    def LDDE_cosine(y_logits, x, edge):
        """Baseline: original LDDE with cosine similarity."""
        distance_x = F.cosine_similarity(x[edge[0]], x[edge[1]])
        y_tilde = torch.log(F.threshold(y_logits, MINIMUM, MINIMUM))
        y_tilde = y_tilde - torch.mean(y_tilde, dim=1, keepdim=True)
        y_tilde = y_tilde / torch.std(y_tilde, dim=1, keepdim=True)
        distance_y = F.cosine_similarity(y_tilde[edge[0]], y_tilde[edge[1]])
        return distance_y - distance_x

    def LDDE_l2(y_logits, x, edge):
        """Ablation: Euclidean (L2) distance replaces cosine similarity."""
        distance_x = F.pairwise_distance(x[edge[0]], x[edge[1]], p=2)
        y_tilde = torch.log(F.threshold(y_logits, MINIMUM, MINIMUM))
        y_tilde = y_tilde - torch.mean(y_tilde, dim=1, keepdim=True)
        y_tilde = y_tilde / torch.std(y_tilde, dim=1, keepdim=True)
        distance_y = F.pairwise_distance(y_tilde[edge[0]], y_tilde[edge[1]], p=2)
        return distance_y - distance_x

    def LDDE_NoDistFeature(y_logits, x, edge):
        """Ablation: remove feature-distance term (logit-distance only)."""
        y_tilde = torch.log(F.threshold(y_logits, MINIMUM, MINIMUM))
        y_tilde = y_tilde - torch.mean(y_tilde, dim=1, keepdim=True)
        y_tilde = y_tilde / torch.std(y_tilde, dim=1, keepdim=True)
        distance_y = F.cosine_similarity(y_tilde[edge[0]], y_tilde[edge[1]])
        return distance_y

    def LDDE_NoExpand(y_logits, x, edge):
        """Ablation: remove log-space expansion (raw logits used directly)."""
        distance_x = F.cosine_similarity(x[edge[0]], x[edge[1]])
        distance_y = F.cosine_similarity(y_logits[edge[0]], y_logits[edge[1]])
        return distance_y - distance_x

    # ── Helper: watermark verification with a specific LDDE variant ──────
    def ablation_watermark_verification(model, wm, wk, trigger, ldde_fn):
        model.eval()
        trigger_y = model(trigger.x, trigger.edge_index).softmax(dim=1)
        v = ldde_fn(trigger_y, trigger.x, trigger.edge_index[:, wk]).flatten()
        wme = torch.where(v < 0, 0, 1)
        hms = int((wme == wm).sum()) / len(wm)
        return hms

    # ── Load data & original model ────────────────────────────────────────
    data = load_data(args)
    model_o = torch.load(
        args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm,
        weights_only=False,
    )
    model_o.eval()

    # ── Setup results path ────────────────────────────────────────────────
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'ablation_ldde/'

    # ── Prepare trigger (shared across variants for fair comparison) ──────
    if args.setting == 1:
        if args.paradigm == 'transductive':
            trigger = data
        elif args.paradigm == 'inductive':
            trigger = data[0]
        else:
            raise ValueError('Error: Wrong paradigm!')
    elif args.setting == 2:
        edge_index = (
            CitationFull(root=args.dataset_path, name='CiteSeer')[0]
            .edge_index.detach()
            .clone()
            .to(args.device)
        )
        trigger = trigger_generation(model_o, edge_index, args)
    else:
        raise ValueError('Error: Wrong setting!')

    # ── Generate watermark & key (standard functions use original LDDE) ───
    wm = watermark_string_generation(args)
    wmk = watermark_key_generation(model_o, trigger, args)
    wm_float = wm.to(torch.float32)

    # ── Variant registry ──────────────────────────────────────────────────
    variants = [
        ('LDDE_cosine',       LDDE_cosine),
        ('LDDE_l2',           LDDE_l2),
        ('LDDE_NoDistFeature', LDDE_NoDistFeature),
        ('LDDE_NoExpand',     LDDE_NoExpand),
    ]

    all_results = []

    # ═══════════════════════════════════════════════════════════════════════
    # Iterative testing loop — one fresh training run per variant
    # ═══════════════════════════════════════════════════════════════════════
    for variant_name, ldde_fn in variants:
        print(f'\n{"=" * 60}')
        print(f'  Ablation variant: {variant_name}')
        print(f'{"=" * 60}')

        # Fresh model copy for this variant
        model_w = copy.deepcopy(model_o)
        optimizer_model = torch.optim.Adam(
            model_w.parameters(), lr=args.wm_lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_model, T_max=128, eta_min=1e-5
        )

        final_epoch = 0
        for epoch in range(args.max_epochs):
            model_w.train()
            optimizer_model.zero_grad()

            trigger_logits = model_w(trigger.x, trigger.edge_index)

            # Task fidelity loss
            if args.paradigm == 'transductive':
                y = model_w(data.x, data.edge_index)
                loss1 = F.cross_entropy(y[data.train_mask], data.y[data.train_mask])
            elif args.paradigm == 'inductive':
                y = model_w(data[0].x, data[0].edge_index)
                loss1 = F.cross_entropy(y, data[0].y)
            else:
                raise ValueError('Error: Wrong paradigm!')

            # Watermark loss — computed with this variant's LDDE
            v = ldde_fn(
                trigger_logits.softmax(dim=1),
                trigger.x,
                trigger.edge_index[:, wmk],
            )
            loss2 = F.binary_cross_entropy_with_logits(v.flatten(), wm_float)
            loss = loss1 + args.coe * loss2

            loss.backward()
            optimizer_model.step()
            scheduler.step()

            if epoch % 20 == 0:
                train_acc, test_acc = test(model_w, data, args)
                bcr = ablation_watermark_verification(
                    model_w, wm, wmk, trigger, ldde_fn
                )
                print(
                    f'  [{variant_name}] Epoch {epoch:3d}: '
                    f'Loss1={loss1:.4f}  Loss2={loss2:.4f}  '
                    f'Train={train_acc:.4f}  Test={test_acc:.4f}  HMS={bcr:.4f}'
                )
                if (bcr > 0.99) and (epoch >= 100):
                    final_epoch = epoch
                    break
            final_epoch = epoch

        # ── Final evaluation for this variant ──────────────────────────
        model_w.eval()
        _, test_acc = test(model_w, data, args)
        hms_w = ablation_watermark_verification(
            model_w, wm, wmk, trigger, ldde_fn
        )

        # HMS on original (unwatermarked) model
        y_o = model_o(trigger.x, trigger.edge_index).softmax(dim=1)
        v_o = ldde_fn(y_o, trigger.x, trigger.edge_index[:, wmk]).flatten()
        hms_o = 1 - hamming_loss(
            wm.detach().cpu().numpy(), (v_o > 0).int().detach().cpu().numpy()
        )

        print(
            f'  [{variant_name}] FINAL: '
            f'Test_acc={test_acc:.4f}  HMS_w={hms_w:.4f}  HMS_o={hms_o:.4f}  '
            f'Epochs={final_epoch + 1}'
        )

        all_results.append(
            {
                'Variant': variant_name,
                'Test_Acc': round(test_acc, 6),
                'HMS_w': round(hms_w, 6),
                'HMS_o': round(hms_o, 6),
                'Epochs': final_epoch + 1,
            }
        )

        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # Save & print comparison table
    # ═══════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(all_results)
    filename = (
        args.results_path
        + args.dataset
        + '_'
        + args.paradigm
        + '_setting'
        + str(args.setting)
        + '_ablation.csv'
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)

    print(f'\n{"=" * 60}')
    print('  Ablation study complete.  Results:')
    print(f'{"=" * 60}')
    print(df.to_string(index=False))
    print(f'\n  Saved to: {filename}')

    # Restore original results path
    args.results_path = results_path


def ldde_distribution(args):
    """Train 5 independent models and 5 watermarked models, then record
    LDDE values across ALL edges of the validation graph for each model."""
    from utils.utils import train
    from utils.models import load_model
    from torch_geometric.utils import subgraph
    from torch_geometric.data import Data

    data = load_data(args)
    model_o = torch.load(
        args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm,
        weights_only=False,
    )
    model_o.eval()

    # ── Determine number of features / classes for fresh models ─────────
    if args.paradigm == 'transductive':
        num_features = data.num_features
        num_classes = data.y.max().item() + 1
    elif args.paradigm == 'inductive':
        num_features = data[0].num_features
        num_classes = data[0].y.max().item() + 1
    else:
        raise ValueError('Error: Wrong paradigm!')

    # ── Setup results path ──────────────────────────────────────────────
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'ldde_distribution/'

    # ── Prepare trigger (still needed for watermark key / embedding) ────
    if args.setting == 1:
        if args.paradigm == 'transductive':
            trigger = data
        elif args.paradigm == 'inductive':
            trigger = data[0]
        else:
            raise ValueError('Error: Wrong paradigm!')
    elif args.setting == 2:
        edge_index = (
            CitationFull(root=args.dataset_path, name='CiteSeer')[0]
            .edge_index.detach()
            .clone()
            .to(args.device)
        )
        trigger = trigger_generation(model_o, edge_index, args)
    else:
        raise ValueError('Error: Wrong setting!')

    # ── Extract validation graph (LDDE is computed on the val graph) ────
    if args.paradigm == 'transductive':
        val_edge_index, _ = subgraph(data.val_mask, data.edge_index,
                                     relabel_nodes=True)
        val_graph = Data(
            x=data.x[data.val_mask],
            y=data.y[data.val_mask],
            edge_index=val_edge_index,
        ).to(args.device)
    elif args.paradigm == 'inductive':
        val_graph = data[1]
    else:
        raise ValueError('Error: Wrong paradigm!')

    wmk = watermark_key_generation(model_o, trigger, args)

    all_ldde_records = {}
    all_wmedge_ldde_records = {}

    # ═════════════════════════════════════════════════════════════════════
    # Train 10 models  —  even index = independent, odd index = watermarked
    # ═════════════════════════════════════════════════════════════════════
    n_independent = 5
    n_watermarked = 5
    n_total = n_independent + n_watermarked
    ind_count = 0
    wm_count = 0

    for i in range(n_total):
        if i % 2 == 0:
            # ── Even index → Independent base model (no watermark) ──────
            ind_count += 1
            print(f'\n{"=" * 50}')
            print(f'  Training independent model {ind_count}/{n_independent}'
                  f'  (model_id={i})')
            print(f'{"=" * 50}')

            torch.manual_seed(i * 100 + 42)
            model_ind = load_model(num_features, num_classes, args)
            optimizer = torch.optim.Adam(
                model_ind.parameters(), lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=128, eta_min=1e-5,
            )
            for epoch in range(args.epochs):
                loss = train(model_ind, data, optimizer, args)
                scheduler.step()
                if epoch % 100 == 0:
                    train_acc, test_acc = test(model_ind, data, args)
                    print(
                        f'  [Ind {ind_count}] Epoch {epoch:3d}: '
                        f'Loss={loss:.4f}  '
                        f'Train={train_acc:.4f}  Test={test_acc:.4f}'
                    )

            model_ind.eval()
            y_ind = model_ind(val_graph.x,
                              val_graph.edge_index).softmax(dim=1)
            # LDDE across ALL edges of validation graph
            v_ind = LDDE(y_ind, val_graph.x, val_graph.edge_index)
            ldde_vals = v_ind.flatten().detach().cpu().numpy()
            all_ldde_records[i] = ldde_vals
            # Watermark edges LDDE (subset via wmk on trigger graph)
            y_ind_trigger = model_ind(trigger.x, trigger.edge_index).softmax(dim=1)
            v_ind_wmedge = LDDE(y_ind_trigger, trigger.x, trigger.edge_index[:, wmk])
            wmedge_ldde_vals = v_ind_wmedge.flatten().detach().cpu().numpy()
            all_wmedge_ldde_records[i] = wmedge_ldde_vals
            torch.cuda.empty_cache()
        else:
            # ── Odd index → Watermarked model ───────────────────────────
            wm_count += 1
            print(f'\n{"=" * 50}')
            print(f'  Training watermarked model {wm_count}/{n_watermarked}'
                  f'  (model_id={i})')
            print(f'{"=" * 50}')

            torch.manual_seed(i * 200 + 123)
            wm = watermark_string_generation(args)
            if args.setting == 1:
                model_w = watermark_embedding_1(
                    copy.deepcopy(model_o), data, wm, wmk, trigger, args,
                )
            elif args.setting == 2:
                model_w = watermark_embedding_2(
                    copy.deepcopy(model_o), data, wm, wmk, trigger, args,
                )
            else:
                raise ValueError('Error: Wrong setting!')

            model_w.eval()
            y_w = model_w(val_graph.x,
                          val_graph.edge_index).softmax(dim=1)
            # LDDE across ALL edges of validation graph
            v_w = LDDE(y_w, val_graph.x, val_graph.edge_index)
            ldde_vals = v_w.flatten().detach().cpu().numpy()
            all_ldde_records[i] = ldde_vals
            # Watermark edges LDDE (subset via wmk on trigger graph)
            y_w_trigger = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
            v_w_wmedge = LDDE(y_w_trigger, trigger.x, trigger.edge_index[:, wmk])
            wmedge_ldde_vals = v_w_wmedge.flatten().detach().cpu().numpy()
            all_wmedge_ldde_records[i] = wmedge_ldde_vals
            torch.cuda.empty_cache()

    # ── Save results ────────────────────────────────────────────────────
    df = pd.DataFrame(all_ldde_records)
    filename = (
        args.results_path
        + args.dataset
        + '_'
        + args.paradigm
        + '_setting'
        + str(args.setting)
        + '_ldde_distribution.csv'
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

    print(f'\n  LDDE distribution saved to: {filename}')
    print(f'  Total records: {len(df)}  (rows per model)')
    n_independent_cols = sum(1 for c in df.columns if int(c) % 2 == 0)
    n_watermarked_cols = sum(1 for c in df.columns if int(c) % 2 != 0)
    print(f'  Independent models: {n_independent_cols}')
    print(f'  Watermarked models: {n_watermarked_cols}')

    # ── Save watermark edges LDDE results ───────────────────────────────
    df_wmedge = pd.DataFrame(all_wmedge_ldde_records)
    filename_wmedge = (
        args.results_path
        + args.dataset
        + '_'
        + args.paradigm
        + '_setting'
        + str(args.setting)
        + '_ldde_distribution_wmedges.csv'
    )
    os.makedirs(os.path.dirname(filename_wmedge), exist_ok=True)
    df_wmedge.to_csv(filename_wmedge, index=False)

    print(f'\n  Watermark edges LDDE distribution saved to: {filename_wmedge}')
    print(f'  Total records: {len(df_wmedge)}  (rows per model)')

    # ── Topological feature extraction: Betweenness Centrality ──────────
    print(f'\n  Computing edge betweenness centrality on validation graph...')

    # Convert PyG data object to NetworkX undirected graph
    G = nx.Graph()
    edge_list = val_graph.edge_index.cpu().t().tolist()
    G.add_edges_from(edge_list)
    print(f'  NetworkX graph: {G.number_of_nodes()} nodes, '
          f'{G.number_of_edges()} unique undirected edges')

    # Calculate Betweenness Centrality for all edges
    edge_bc = nx.edge_betweenness_centrality(G)

    # Map centrality scores back to each PyG edge_index entry.
    # NetworkX stores undirected edges as sorted (min, max) tuples.
    num_edges = val_graph.edge_index.size(1)
    centrality_scores = np.zeros(num_edges)
    for e in range(num_edges):
        u = val_graph.edge_index[0, e].item()
        v = val_graph.edge_index[1, e].item()
        key = (min(u, v), max(u, v))
        centrality_scores[e] = edge_bc.get(key, 0.0)

    # Identify edges with highest 5% and lowest 5% centrality scores
    n_edges = len(centrality_scores)
    n_top = max(1, int(np.ceil(n_edges * 0.05)))
    n_bottom = max(1, int(np.ceil(n_edges * 0.05)))

    sorted_indices = np.argsort(centrality_scores)
    bottom_5_indices = sorted_indices[:n_bottom]   # lowest 5% centrality
    top_5_indices = sorted_indices[-n_top:]         # highest 5% centrality

    # Mark these edges using boolean tensors
    top_5_mask = torch.zeros(num_edges, dtype=torch.bool)
    bottom_5_mask = torch.zeros(num_edges, dtype=torch.bool)
    top_5_mask[top_5_indices] = True
    bottom_5_mask[bottom_5_indices] = True

    print(f'  Top    5% edges: {top_5_mask.sum().item()} '
          f'(centrality range: {centrality_scores[top_5_indices].min():.6f} - '
          f'{centrality_scores[top_5_indices].max():.6f})')
    print(f'  Bottom 5% edges: {bottom_5_mask.sum().item()} '
          f'(centrality range: {centrality_scores[bottom_5_indices].min():.6f} - '
          f'{centrality_scores[bottom_5_indices].max():.6f})')

    # Extract corresponding LDDE values for top / bottom 5% subsets
    top_ldde_records = {}
    bottom_ldde_records = {}
    for mdl_id, ldde_vals in all_ldde_records.items():
        top_ldde_records[mdl_id] = ldde_vals[top_5_indices]
        bottom_ldde_records[mdl_id] = ldde_vals[bottom_5_indices]

    # Save top 5% betweenness-centrality LDDE subset
    df_top = pd.DataFrame(top_ldde_records)
    filename_top = (
        args.results_path
        + args.dataset + '_' + args.paradigm + '_setting'
        + str(args.setting) + '_ldde_distribution_top5_bc.csv'
    )
    os.makedirs(os.path.dirname(filename_top), exist_ok=True)
    df_top.to_csv(filename_top, index=False)
    print(f'\n  Top 5% BC LDDE distribution saved to: {filename_top}')
    print(f'  Total records: {len(df_top)}  (rows per model)')

    # Save bottom 5% betweenness-centrality LDDE subset
    df_bottom = pd.DataFrame(bottom_ldde_records)
    filename_bottom = (
        args.results_path
        + args.dataset + '_' + args.paradigm + '_setting'
        + str(args.setting) + '_ldde_distribution_bottom5_bc.csv'
    )
    os.makedirs(os.path.dirname(filename_bottom), exist_ok=True)
    df_bottom.to_csv(filename_bottom, index=False)
    print(f'\n  Bottom 5% BC LDDE distribution saved to: {filename_bottom}')
    print(f'  Total records: {len(df_bottom)}  (rows per model)')

    args.results_path = results_path


def trigger_ablation(args):
    """Ablation study over trigger graph structures.

    For Setting 2 the trigger graph defaults to CiteSeer.  This function
    extends the evaluation to Roman‑empire, ER (Erdős‑Rényi), and BA
    (Barabási‑Albert) so the impact of the trigger‑graph topology can be
    quantified.
    """
    from torch_geometric.utils import erdos_renyi_graph, barabasi_albert_graph
    from torch_geometric.datasets import HeterophilousGraphDataset

    # ── Load data & original model ──────────────────────────────────────
    data = load_data(args)
    model_o = torch.load(
        args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm,
        weights_only=False,
    )
    model_o.eval()

    # ── Trigger ablation is only meaningful for Setting 2 ───────────────
    if args.setting != 2:
        print(
            'Trigger ablation is designed for Setting 2 '
            '(external trigger graph).  Skipping.'
        )
        return

    # ── Setup results path ──────────────────────────────────────────────
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'trigger_ablation/'

    # ── Build registry of graph structures ──────────────────────────────
    # Use CiteSeer as the reference to match size for ER / BA
    edge_index_ref = (
        CitationFull(root=args.dataset_path, name='CiteSeer')[0]
        .edge_index.detach()
        .clone()
        .to(args.device)
    )
    num_nodes_ref = int(edge_index_ref.max().item()) + 1
    num_edges_ref = edge_index_ref.size(1) // 2  # undirected → half the entries

    graph_structures = {}

    # 1.  CiteSeer (baseline)
    graph_structures['CiteSeer'] = edge_index_ref

    # 2.  Roman‑empire
    try:
        edge_index_roman = (
            HeterophilousGraphDataset(
                root=args.dataset_path, name='Roman-empire'
            )[0]
            .edge_index.detach()
            .clone()
            .to(args.device)
        )
        graph_structures['Roman-empire'] = edge_index_roman
    except Exception as e:
        print(f'  [WARNING] Could not load Roman-empire: {e}')

    # 3.  ER (Erdős‑Rényi) – matched node count and approximate edge count
    p_er = 2.0 * num_edges_ref / (num_nodes_ref * (num_nodes_ref - 1))
    edge_index_er = erdos_renyi_graph(num_nodes_ref, p_er).to(args.device)
    graph_structures['ER'] = edge_index_er

    # 4.  BA (Barabási‑Albert) – matched node count, attachment edges = 1
    edge_index_ba = barabasi_albert_graph(num_nodes_ref, 1).to(args.device)
    graph_structures['BA'] = edge_index_ba

    # ═════════════════════════════════════════════════════════════════════
    # Iterate over graph structures
    # ═════════════════════════════════════════════════════════════════════
    all_results = []

    for graph_name, edge_index in graph_structures.items():
        print(f'\n{"=" * 60}')
        print(f'  Trigger graph: {graph_name}')
        print(f'  Nodes: {int(edge_index.max().item()) + 1}  '
              f'Edges (directed entries): {edge_index.size(1)}')
        print(f'{"=" * 60}')

        # Generate trigger on this graph structure
        trigger = trigger_generation(model_o, edge_index, args)
        print(f'  Total trigger edges (edge_attr=True): {trigger.edge_attr.sum().item()}')

        # Generate watermark & key on this trigger
        wm = watermark_string_generation(args)
        wmk = watermark_key_generation(model_o, trigger, args)
        wm_float = wm.to(torch.float32)

        # Embed watermark
        model_w = watermark_embedding_2(
            copy.deepcopy(model_o), data, wm, wmk, trigger, args
        )
        model_w.eval()

        # ── Evaluation ──────────────────────────────────────────────────
        _, test_acc = test(model_w, data, args)
        hms_w = watermark_verification(model_w, wm, wmk, trigger)

        # HMS on the original (unwatermarked) model for this trigger
        y_o = model_o(trigger.x, trigger.edge_index).softmax(dim=1)
        v_o = LDDE(y_o, trigger.x, trigger.edge_index[:, wmk]).flatten()
        hms_o = 1 - hamming_loss(
            wm.detach().cpu().numpy(), (v_o > 0).int().detach().cpu().numpy()
        )

        print(
            f'  [{graph_name}] Test_acc={test_acc:.4f}  '
            f'HMS_w={hms_w:.4f}  HMS_o={hms_o:.4f}'
        )

        all_results.append({
            'Graph': graph_name,
            'Num_Nodes': int(edge_index.max().item()) + 1,
            'Num_Edges': edge_index.size(1),
            'Test_Acc': round(test_acc, 6),
            'HMS_w': round(hms_w, 6),
            'HMS_o': round(hms_o, 6),
        })

        torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════════════
    # Save & print comparison table
    # ═════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(all_results)
    filename = (
        args.results_path
        + args.dataset
        + '_'
        + args.paradigm
        + '_setting'
        + str(args.setting)
        + '_trigger_ablation.csv'
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)

    print(f'\n{"=" * 60}')
    print('  Trigger ablation study complete.  Results:')
    print(f'{"=" * 60}')
    print(df.to_string(index=False))
    print(f'\n  Saved to: {filename}')

    args.results_path = results_path



def ablation_finetune_lr(args):
    """Ablation study over fine-tuning learning rates.

    For each learning rate, embed a fresh watermark, run fine-tuning on
    both the watermarked model and an independently trained model, and
    record the final TAC and HMS to quantify how LR affects watermark
    robustness under fine-tuning attacks.
    """
    from watermark.robust import fine_tuning

    # ── Load data & original model ──────────────────────────────────────
    data = load_data(args)
    model_o = torch.load(
        args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm,
        weights_only=False,
    )
    model_o.eval()

    # ── Setup results path ──────────────────────────────────────────────
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'ablation_finetune_lr/'

    # ── Prepare trigger (shared across all LR values) ───────────────────
    if args.setting == 1:
        if args.paradigm == 'transductive':
            trigger = data
        elif args.paradigm == 'inductive':
            trigger = data[0]
        else:
            raise ValueError('Error: Wrong paradigm!')
    elif args.setting == 2:
        edge_index = (
            CitationFull(root=args.dataset_path, name='CiteSeer')[0]
            .edge_index.detach()
            .clone()
            .to(args.device)
        )
        trigger = trigger_generation(model_o, edge_index, args)
    else:
        raise ValueError('Error: Wrong setting!')

    lr_values = [5e-4, 1e-4, 5e-5, 1e-5]
    all_results = []

    # ═════════════════════════════════════════════════════════════════════
    # Iterate over learning rates — fresh watermark embedding per value
    # ═════════════════════════════════════════════════════════════════════
    for lr in lr_values:
        print(f'\n{"=" * 60}')
        print(f'  Fine-tuning LR: {lr}')
        print(f'{"=" * 60}')

        # Embed watermark on a fresh model copy
        model_w, wm, wmk, trigger_w, model_ind = setting(
            copy.deepcopy(model_o), copy.deepcopy(model_o), data, args,
        )
        model_w.eval()

        # ── Fine-tune the watermarked model ──────────────────────────
        _, tac_list_w, hms_list_w = fine_tuning(
            copy.deepcopy(model_w), data, wm, wmk, trigger_w, args, lr=lr,
        )
        final_tac_w = tac_list_w[-1]
        final_hms_w = hms_list_w[-1]

        print(f'  [LR={lr}] Watermarked  ->  TAC={final_tac_w:.4f}  '
              f'HMS={final_hms_w:.4f}')

        result = {
            'LR': lr,
            'TAC_w': round(final_tac_w, 6),
            'HMS_w': round(final_hms_w, 6),
        }

        # ── Fine-tune the independent model (control) ─────────────────
        if model_ind is not None:
            model_ind.eval()
            _, tac_list_i, hms_list_i = fine_tuning(
                copy.deepcopy(model_ind), data, wm, wmk, trigger_w, args, lr=lr,
            )
            final_tac_i = tac_list_i[-1]
            final_hms_i = hms_list_i[-1]

            print(f'  [LR={lr}] Independent ->  TAC={final_tac_i:.4f}  '
                  f'HMS={final_hms_i:.4f}')
            result['TAC_i'] = round(final_tac_i, 6)
            result['HMS_i'] = round(final_hms_i, 6)

        all_results.append(result)
        torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════════════
    # Save & print comparison table
    # ═════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(all_results)
    filename = (
        args.results_path
        + args.dataset
        + '_'
        + args.paradigm
        + '_setting'
        + str(args.setting)
        + '_ablation_finetune_lr.csv'
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)

    print(f'\n{"=" * 60}')
    print('  Fine-tuning LR ablation complete.  Results:')
    print(f'{"=" * 60}')
    print(df.to_string(index=False))
    print(f'\n  Saved to: {filename}')

    # Restore original results path
    args.results_path = results_path



def ablation_coe(args):
    """Ablation study over the trigger-generation loss2 regularization coefficient.

    For each coefficient value, generate a fresh trigger graph (where the
    coefficient balances LDDE-minimisation against feature-diversity
    regularisation inside trigger_generation), embed a watermark, and
    record the resulting TAC and HMS.

    This ablation is specific to Setting 2 — the coefficient only affects
    trigger_generation, which is not used in Setting 1.
    """

    # ── This ablation is only meaningful for Setting 2 ──────────────────
    if args.setting != 2:
        print(
            'Trigger loss2 coefficient ablation is designed for Setting 2 '
            '(external trigger graph).  Skipping.'
        )
        return

    # ── Load data & original model ──────────────────────────────────────
    data = load_data(args)
    model_o = torch.load(
        args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm,
        weights_only=False,
    )
    model_o.eval()

    # ── Setup results path ──────────────────────────────────────────────
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'ablation_coe/'

    # ── Shared edge_index for trigger generation ────────────────────────
    edge_index = (
        CitationFull(root=args.dataset_path, name='CiteSeer')[0]
        .edge_index.detach()
        .clone()
        .to(args.device)
    )

    coe_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    all_results = []

    # ═════════════════════════════════════════════════════════════════════
    # Iterate over coefficient values — fresh trigger + embedding per value
    # ═════════════════════════════════════════════════════════════════════
    for coe in coe_values:
        print(f'\n{"=" * 60}')
        print(f'  Trigger loss2 coefficient: {coe}')
        print(f'{"=" * 60}')

        # Temporarily override the trigger loss2 coefficient
        original_coe = getattr(args, 'trigger_loss2_coe', 1e-4)
        args.trigger_loss2_coe = coe

        # Generate trigger with this coefficient
        trigger = trigger_generation(model_o, edge_index, args)

        # Generate watermark string and key on this trigger
        wm = watermark_string_generation(args)
        wmk = watermark_key_generation(model_o, trigger, args)

        # Embed watermark (Setting 2 uses watermark_embedding_2)
        model_w = watermark_embedding_2(
            copy.deepcopy(model_o), data, wm, wmk, trigger, args,
        )
        model_w.eval()

        _, test_acc = test(model_w, data, args)
        hms = watermark_verification(model_w, wm, wmk, trigger)

        print(f'  [Coe={coe}] TAC={test_acc:.4f}  HMS={hms:.4f}')

        all_results.append({
            'Coefficient': coe,
            'TAC': round(test_acc, 6),
            'HMS': round(hms, 6),
        })

        # Restore the original coefficient
        args.trigger_loss2_coe = original_coe
        torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════════════
    # Save & print comparison table
    # ═════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(all_results)
    filename = (
        args.results_path
        + args.dataset
        + '_'
        + args.paradigm
        + '_setting'
        + str(args.setting)
        + '_ablation_coe.csv'
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)

    print(f'\n{"=" * 60}')
    print('  Trigger loss2 coefficient ablation complete.  Results:')
    print(f'{"=" * 60}')
    print(df.to_string(index=False))
    print(f'\n  Saved to: {filename}')

    # Restore original results path
    args.results_path = results_path



def collusion_attack(args):
    data = load_data(args)
    model_o = torch.load(
        args.model_path + args.dataset + '/' + args.model + '_' + args.paradigm,
        weights_only=False,
    )
    model_o.eval()
    results_path = copy.deepcopy(args.results_path)
    args.results_path = results_path + 'collusion_attack/'

    # Helper: select embedding function based on setting (reuse existing functions)
    def _embed(model, data, wm, wk, trigger, args):
        if args.setting == 1:
            return watermark_embedding_1(model, data, wm, wk, trigger, args)
        elif args.setting == 2:
            return watermark_embedding_2(model, data, wm, wk, trigger, args)
        else:
            raise ValueError('Error: Wrong setting!')

    # ========================================================================
    # Phase 1: Full-Key Watermarking Baseline
    # Generate 4 watermark models, each with the full watermark key.
    # During verification, average model outputs across all 4 models,
    # then compute HMS from the averaged signal.
    # ========================================================================
    print('=' * 60)
    print('Phase 1: Full-Key Watermarking Baseline')
    print('=' * 60)

    models_w_list = []
    models_i_list = []
    wm_list = []
    wmk_list = []
    trigger_list = []
    for i in range(4):
        model_w, wm, wmk, trigger, model_i = setting(copy.deepcopy(model_o), copy.deepcopy(model_o), data, args)
        models_w_list.append(model_w)
        models_i_list.append(model_i)
        wm_list.append(wm)
        wmk_list.append(wmk)
        trigger_list.append(trigger)
        torch.cuda.empty_cache()
        print(f'Phase 1 No.{i} model generated')

    # Verify wmk consistency across all models
    first_wmk = wmk_list[0]
    for i, t in enumerate(wmk_list[1:], 1):
        if not torch.equal(first_wmk, t):
            print(f'Warning: wmk[{i}] differs from wmk[0]!')
    wmk = first_wmk.clone()

    # Verify trigger consistency across all models
    first_trigger = trigger_list[0]
    for i, t in enumerate(trigger_list[1:], 1):
        if not (torch.equal(first_trigger.x, t.x) and
                torch.equal(first_trigger.edge_index, t.edge_index)):
            print(f'Warning: trigger[{i}] differs from trigger[0]!')
    trigger = first_trigger.clone()

    y_list = []
    if args.paradigm == 'transductive':
        for m in models_w_list:
            m.eval()
            y_list.append(m(data.x, data.edge_index))
        y_pred_1 = torch.stack(y_list).mean(dim=0).softmax(dim=1).argmax(dim=1)
        tac_1 = int((y_pred_1[data.test_mask] == data.y[data.test_mask]).sum()) / len(data.y[data.test_mask])
    elif args.paradigm == 'inductive':
        for m in models_w_list:
            m.eval()
            y_list.append(m(data[2].x, data[2].edge_index))
        y_pred_1 = torch.stack(y_list).mean(dim=0).softmax(dim=1).argmax(dim=1)
        tac_1 = int((y_pred_1 == data[2].y).sum()) / len(data[2].y)
    else:
        raise ValueError('Error: Wrong paradigm!')

    # Collusion evaluation: average logits across all 4 models, then verify
    y_list = []
    for m in models_w_list:
        m.eval()
        y_list.append(m(trigger.x, trigger.edge_index))
    y_avg = torch.stack(y_list).mean(dim=0).softmax(dim=1)
    v = LDDE(y_avg, trigger.x, trigger.edge_index[:, wmk]).flatten()
    wme = torch.where(v < 0, 0, 1)
    hms_w = []
    for i in range(4):
        hms = int((wme == wm_list[i]).sum()) / len(wm_list[i])
        hms_w.append(hms)
    print('Phase 1 HMS_w (Full Key):', hms_w)
    
    y_list = []
    for m in models_i_list:
        m.eval()
        y_list.append(m(trigger.x, trigger.edge_index))
    y_avg = torch.stack(y_list).mean(dim=0).softmax(dim=1)
    v = LDDE(y_avg, trigger.x, trigger.edge_index[:, wmk]).flatten()
    wme = torch.where(v < 0, 0, 1)
    hms_i = []
    for i in range(4):
        hms = int((wme == wm_list[i]).sum()) / len(wm_list[i])
        hms_i.append(hms)
    print('Phase 1 HMS_i (Full Key):', hms_i)
    
    # ========================================================================
    # Save results to CSV
    # ========================================================================
    # CSV file path
    filename = args.results_path + args.dataset + '_' + args.paradigm + '_setting' + str(
        args.setting) + '_' + 'collusion_attack_p1.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    headers = ['TAC', 'HMS_w1', 'HMS_w2', 'HMS_w3', 'HMS_w4', 'HMS_i1', 'HMS_i2', 'HMS_i3', 'HMS_i4']
    file_exists = os.path.isfile(filename)
    if not file_exists:
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([tac_1] + hms_w + hms_i)

    # ========================================================================
    # Phase 2: Non-Overlapping Key Partitioning
    # Split the watermark key into 4 equal, non-overlapping segments.
    # Generate 4 new watermark models, each assigned one unique segment.
    # During verification, average model outputs and verify against the
    # FULL key to test collusion effectiveness.
    # ========================================================================
    print('=' * 60)
    print('Phase 2: Non-Overlapping Key Partitioning')
    print('=' * 60)

    # Generate one base set: wm, wmk, trigger
    _, wm, wmk, trigger, _ = setting(copy.deepcopy(model_o), None, data, args)

    # Partition the watermark key into 4 equal, non-overlapping subsets
    # true_indices: positions where wmk is True, in the order produced by
    # watermark_key_generation (sorted by LDDE magnitude, smallest first)
    true_indices = torch.where(wmk)[0]
    n_wm_total = len(true_indices)
    part_size = n_wm_total // 4

    # Verify divisibility
    assert n_wm_total % 4 == 0, \
        f'n_wm ({n_wm_total}) must be divisible by 4 for equal partitioning'

    wmk_list = []
    for i in range(4):
        start = i * part_size
        end = (i + 1) * part_size
        part_indices = true_indices[start:end]

        # Create boolean mask: same length as wmk, True only at part_indices
        wmk_i = torch.zeros_like(wmk, dtype=torch.bool)
        wmk_i[part_indices] = True
        wmk_list.append(wmk_i)

    wm_list = torch.split(wm, part_size)
    # Generate 4 models, each with one non-overlapping key partition
    models_w_list = []
    models_i_list = []
    for i in range(4):
        model_w = _embed(copy.deepcopy(model_o), data, wm_list[i],
                           wmk_list[i], trigger, args)
        models_w_list.append(model_w)
        model_i = _embed(copy.deepcopy(model_o), data, torch.randint(0, 2, (part_size,), dtype=torch.uint8).to(args.device),
                           wmk_list[i], trigger, args)
        models_i_list.append(model_i)
        torch.cuda.empty_cache()
        print(f'Phase 2 No.{i} partitioned model generated')

    y_list = []
    if args.paradigm == 'transductive':
        for m in models_w_list:
            m.eval()
            y_list.append(m(data.x, data.edge_index))
        y_pred_2 = torch.stack(y_list).mean(dim=0).softmax(dim=1).argmax(dim=1)
        tac_2 = int((y_pred_2[data.test_mask] == data.y[data.test_mask]).sum()) / len(data.y[data.test_mask])
    elif args.paradigm == 'inductive':
        for m in models_w_list:
            m.eval()
            y_list.append(m(data[2].x, data[2].edge_index))
        y_pred_2 = torch.stack(y_list).mean(dim=0).softmax(dim=1).argmax(dim=1)
        tac_2 = int((y_pred_2 == data[2].y).sum()) / len(data[2].y)
    else:
        raise ValueError('Error: Wrong paradigm!')
    
    # Average outputs across all 4 partitioned models
    y_list = []
    for m in models_w_list:
        m.eval()
        y_list.append(m(trigger.x, trigger.edge_index))
    y_avg = torch.stack(y_list).mean(dim=0).softmax(dim=1)
    # Individual HMS for each partitioned model as baseline
    hms_w = []
    for i in range(4):
        v = LDDE(y_avg, trigger.x, trigger.edge_index[:, wmk_list[i]]).flatten()
        wme = torch.where(v < 0, 0, 1)
        hms = int((wme == wm_list[i]).sum()) / len(wm_list[i])
        hms_w.append(hms)
    print(f'Phase 2 HMS_w (partitioned key): {hms_w}')

    y_list = []
    for m in models_i_list:
        m.eval()
        y_list.append(m(trigger.x, trigger.edge_index))
    y_avg = torch.stack(y_list).mean(dim=0).softmax(dim=1)
    # Individual HMS for each partitioned model as baseline
    hms_i = []
    for i in range(4):
        v = LDDE(y_avg, trigger.x, trigger.edge_index[:, wmk_list[i]]).flatten()
        wme = torch.where(v < 0, 0, 1)
        hms = int((wme == wm_list[i]).sum()) / len(wm_list[i])
        hms_i.append(hms)
    print(f'Phase 2 HMS_i (partitioned key): {hms_i}')

    # ========================================================================
    # Save results to CSV
    # ========================================================================
    filename = args.results_path + args.dataset + '_' + args.paradigm + '_setting' + str(
        args.setting) + '_' + 'collusion_attack_p2.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    headers = ['TAC', 'HMS_w1', 'HMS_w2', 'HMS_w3', 'HMS_w4', 'HMS_i1', 'HMS_i2', 'HMS_i3', 'HMS_i4']
    file_exists = os.path.isfile(filename)
    if not file_exists:
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([tac_2] + hms_w + hms_i)

    args.results_path = results_path



def assess_insight(args):
    # insight2(args)
    # insight3(args)
    # multibit(args)
    # watermark_collision(args)
    # for i in range(25):
    #     hard_label(args)
    # ablation_coe(args)
    # ablation_finetune_lr(args)
    ldde_distribution(args)
    # trigger_ablation(args)
    # for i in range(25):
    #     ablation_ldde(args)
    # collusion_attack(args)
