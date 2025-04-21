import copy
import csv
import os.path

import numpy as np
from torch_geometric.utils import subgraph
from torch_geometric.datasets import Planetoid
from watermark.watermark import *
from watermark.robust import *
from utils.utils import test
from utils.models import SAGE, SSG


def setting(model_o, model_ind, train_data, val_data, test_data, args):
    if args.setting == 1:
        return setting1(copy.deepcopy(model_o), copy.deepcopy(model_ind), train_data, val_data, test_data, args)
    elif args.setting == 2:
        return setting2(copy.deepcopy(model_o), copy.deepcopy(model_ind), train_data, val_data, test_data, args)
    elif args.setting == 3:
        return setting3(copy.deepcopy(model_o), copy.deepcopy(model_ind), train_data, val_data, test_data, args)
    else:
        print('error')
        exit()


def setting1(model_o, model_ind, train_data, val_data, test_data, args):
    filename = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'setting1.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    headers = ['Mo train acc', 'Mo val acc', 'Mo test acc', 'Mo misclass', 'Mi train acc', 'Mi val acc', 'Mi test acc', 'Mi hms', 'Mw train acc', 'Mw val acc', 'Mw test acc', 'Mw hms', 'Time', 'Mw misclass']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    # assess before embedding watermark
    train_acc = test(model_o, train_data)
    val_acc = test(model_o, val_data)
    test_acc = test(model_o, test_data)
    y = model_o(test_data.x, test_data.edge_index).argmax(dim=1)
    misclass = ((y == 2) & (test_data.y != 2)).sum() / (test_data.y != 2).sum().item()

    print(f'Accuracy on the original model: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    print(f'misclassification rate: {misclass:.4f}')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {misclass:.4f}, ')

    # embedding watermark
    trigger = train_data
    start_time = time.time()
    wm = watermark_string_generation(args)
    wmk = watermark_key_generation(model_o, trigger, wm, args)
    model_w = watermark_embedding_1(model_o, train_data, val_data, test_data, wm, wmk, trigger, args)
    end_time = time.time()
    sub_time = end_time - start_time

    # acc and bcr of model_ind
    print('Independently trained model watermark embedding')
    wm_ind = watermark_string_generation(args)
    trigger_ind = train_data
    wmk_ind = watermark_key_generation(model_ind, trigger_ind, wm_ind, args)
    model_ind = watermark_embedding_1(model_ind, train_data, val_data, test_data, wm_ind, wmk_ind, trigger_ind, args)
    hms = watermark_verification(model_ind, wm, wmk, trigger)
    print(f'HMS of the independently trained model :{hms:.4f}')
    train_acc = test(model_ind, train_data)
    val_acc = test(model_ind, val_data)
    test_acc = test(model_ind, test_data)
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f}, ')

    # acc and bcr of model_wm
    hms = watermark_verification(model_w, wm, wmk, trigger)
    train_acc = test(model_w, train_data)
    val_acc = test(model_w, val_data)
    test_acc = test(model_w, test_data)
    y = model_w(test_data.x, test_data.edge_index).argmax(dim=1)
    misclass = ((y == 2) & (test_data.y != 2)).sum() / (test_data.y != 2).sum().item()
    print(f'Accuracy on Model_w: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    print(f'BCR on Model_w: {hms:.4f}')
    print(f'Time of Model_w: {sub_time:.4f}')
    print(f'misclassification rate: {misclass:.4f}')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f}, {sub_time:.4f}, {misclass:.4f}\n')
    print('--------------------------------------------------')
    return model_w, wm, wmk, trigger, model_ind


def setting2(model_o, model_ind, train_data, val_data, test_data, args):
    filename = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'setting2.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    headers = ['Mo train acc', 'Mo val acc', 'Mo test acc', 'Mo misclass', 'Mi train acc', 'Mi val acc', 'Mi test acc', 'Mi hms',
               'Mw train acc', 'Mw val acc', 'Mw test acc', 'Mw hms', 'Time', 'Mw misclass']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    # assess before embedding watermark
    train_acc = test(model_o, train_data)
    val_acc = test(model_o, val_data)
    test_acc = test(model_o, test_data)
    y = model_o(test_data.x, test_data.edge_index).argmax(dim=1)
    misclass = ((y == 2) & (test_data.y != 2)).sum() / (test_data.y != 2).sum().item()
    print(f'Accuracy on the original model: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {misclass:.4f}, ')

    # embedding watermark
    edge_index = Planetoid(root=args.dataset_path, name='PubMed')[0].edge_index.detach().clone().to(device)
    trigger = trigger_generation(model_o, edge_index, args)
    start_time = time.time()
    wm = watermark_string_generation(args)
    wmk = watermark_key_generation(model_o, trigger, wm, args)
    model_w = watermark_embedding_1(model_o, train_data, val_data, test_data, wm, wmk, trigger, args)
    end_time = time.time()
    sub_time = end_time - start_time

    # acc and bcr of model_ind
    print('Independently trained model watermark inject')
    wm_ind = watermark_string_generation(args)
    wmk_ind = watermark_key_generation(model_ind, trigger, wm_ind, args)
    model_ind = watermark_embedding_1(model_ind, train_data, val_data, test_data, wm_ind, wmk_ind, trigger, args)
    hms = watermark_verification(model_ind.to(device), wm, wmk, trigger)
    print(f'HMS on the independently trained model: {hms:.4f}')
    train_acc = test(model_ind, train_data)
    val_acc = test(model_ind, val_data)
    test_acc = test(model_ind, test_data)
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f}, ')

    # acc and bcr of model_wm
    hms = watermark_verification(model_w, wm, wmk, trigger)
    train_acc = test(model_w, train_data)
    val_acc = test(model_w, val_data)
    test_acc = test(model_w, test_data)
    y = model_w(test_data.x, test_data.edge_index).argmax(dim=1)
    misclass = ((y == 2) & (test_data.y != 2)).sum() / (test_data.y != 2).sum().item()
    print(f'Accuracy on the watermarked model: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    print(f'HMS on Model_w: {hms:.4f}')
    print(f'Time of Model_w: {sub_time:.4f}')
    print(f'misclassification rate: {misclass:.4f}')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f}, {sub_time:.4f}, {misclass:.4f}\n')
    print('--------------------------------------------------')
    return model_w, wm, wmk, trigger, model_ind


def setting3(model_o, model_ind, train_data, val_data, test_data, args):
    filename = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'setting3.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    headers = ['Mo train acc', 'Mo val acc', 'Mo test acc', 'Mo misclass', 'Mi train acc', 'Mi val acc', 'Mi test acc', 'Mi hms',
               'Mw train acc', 'Mw val acc', 'Mw test acc', 'Mw hms', 'Time', 'Mw misclass']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    # assess before embedding watermark
    train_acc = test(model_o, train_data)
    val_acc = test(model_o, val_data)
    test_acc = test(model_o, test_data)
    y = model_o(test_data.x, test_data.edge_index).argmax(dim=1)
    misclass = ((y == 2) & (test_data.y != 2)).sum() / (test_data.y != 2).sum().item()
    print(f'Accuracy on the original model: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    print(f'misclassification rate: {misclass:.4f}')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {misclass:.4f},')

    # embedding watermark
    edge_index = Planetoid(root=args.dataset_path, name='PubMed')[0].edge_index.detach().clone().to(device)
    trigger = trigger_generation(model_o, edge_index, args)
    start_time = time.time()
    wm = watermark_string_generation(args)
    wmk = watermark_key_generation(model_o, trigger, wm, args)
    model_w = watermark_embedding_1(model_o, train_data, val_data, test_data, wm, wmk, trigger, args)
    end_time = time.time()
    sub_time = end_time - start_time

    # acc and bcr of model_ind
    print('Independently trained model watermark inject')
    wm_ind = watermark_string_generation(args)
    wmk_ind = watermark_key_generation(model_ind, trigger, wm_ind, args)
    model_ind = watermark_embedding_2(model_ind, None, val_data, test_data, wm_ind, wmk_ind, trigger, args)
    hms = watermark_verification(model_ind.to(device), wm, wmk, trigger)
    print(f'HMS on the independently trained model: {hms:.4f}')
    train_acc = test(model_ind, train_data)
    val_acc = test(model_ind, val_data)
    test_acc = test(model_ind, test_data)
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f}, ')

    # acc and bcr of model_wm
    hms = watermark_verification(model_w, wm, wmk, trigger)
    train_acc = test(model_w, train_data)
    val_acc = test(model_w, val_data)
    test_acc = test(model_w, test_data)
    y = model_w(test_data.x, test_data.edge_index).argmax(dim=1)
    misclass = ((y == 2) & (test_data.y != 2)).sum() / (test_data.y != 2).sum().item()
    print(f'Accuracy on the watermarked model: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    print(f'HMS on Model_w: {hms:.4f}')
    print(f'Time of Model_w: {sub_time:.4f}')
    print(f'misclassification rate: {misclass:.4f}')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f}, {sub_time:.4f}, {misclass:.4f}\n')
    print('--------------------------------------------------')
    return model_w, wm, wmk, trigger, model_ind


def assess_pruning(model_w, model_ind, train_data, val_data, test_data, wm, wmk, trigger, args):
    filename_w = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'pruning_w.csv'
    headers = [item for i in range(0, 101, 10) for item in [f'{i}%test_acc']] + [item for i in range(0, 101, 10) for item in [f'{i}%hms']]
    if not os.path.isfile(filename_w):
        with open(filename_w, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    tac_list = []
    hms_list = []
    for i in range(11):
        model_copy = model_pruning(copy.deepcopy(model_w), i / 10)
        train_acc = test(model_copy, train_data)
        val_acc = test(model_copy, val_data)
        test_acc = test(model_copy, test_data)
        print(f'Accuracy after pruning: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
        hms = watermark_verification(model_copy, wm, wmk, trigger)
        print(f'HMS with {i / 10} pruning:{hms:.4f}')
        print('------------------------------------')
        tac_list.append(test_acc)
        hms_list.append(hms)

    tac_list = np.round(np.array(tac_list), 2)
    bcr_list = np.round(np.array(hms_list), 3)
    with open(filename_w, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, bcr_list]))

    filename_i = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'pruning_i.csv'
    if not os.path.isfile(filename_i):
        with open(filename_i, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    tac_list = []
    hms_list = []
    for i in range(11):
        model_copy = model_pruning(copy.deepcopy(model_ind), i / 10)
        train_acc = test(model_copy, train_data)
        val_acc = test(model_copy, val_data)
        test_acc = test(model_copy, test_data)
        print(f'Accuracy of Model_ind after pruning of: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
        hms = watermark_verification(model_copy, wm, wmk, trigger)
        print(f'HMS of Model_ind with {i / 10} pruning:{hms:.4f}')
        print('------------------------------------')
        tac_list.append(test_acc)
        hms_list.append(hms)

    tac_list = np.round(np.array(tac_list), 2)
    bcr_list = np.round(np.array(hms_list), 3)
    with open(filename_i, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, bcr_list]))



def assess_fine_tuning(model_w, model_ind, train_data, val_data, test_data, wm, wmk, trigger, args):
    filename = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'fine_tuning_w.csv'
    headers = [item for i in range(201) for item in [f'{i}test_acc']] + [item for i in range(201) for item in [f'{i}hms']]
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    _, tac_list, hms_list =fine_tuning(copy.deepcopy(model_w), val_data, test_data, wm, wmk, trigger, args.lr/2)
    print(f'Accuracy after fine-tuning: Train: {tac_list[-1]:.4f}')
    print(f'HMS after fine-tuning:{hms_list[-1]:.4f}')
    print('------------------------------------')
    tac_list = np.round(np.array(tac_list), 2)
    hms_list = np.round(np.array(hms_list), 3)
    with open(filename, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, hms_list]))

    filename = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'fine_tuning_i.csv'
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    _, tac_list, hms_list = fine_tuning(copy.deepcopy(model_ind), val_data, test_data, wm, wmk, trigger, args.lr / 2)
    print(f'Accuracy of Model_ind after fine-tuning: Train: {tac_list[-1]:.4f}')
    print(f'HMS of Model_ind after fine-tuning:{hms_list[-1]:.4f}')
    print('------------------------------------')
    tac_list = np.round(np.array(tac_list), 2)
    hms_list = np.round(np.array(hms_list), 3)
    with open(filename, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, hms_list]))



def assess_overwriting(model, model_i, train_data, val_data, test_data, wm, wmk, trigger, args):
    model = copy.deepcopy(model)
    filename = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'overwriting.csv'
    headers =  ['train_acc', 'val_acc', 'test_acc', 'before', 'train_acc', 'val_acc', 'test_acc', 'after']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    train_acc = test(model, train_data)
    val_acc = test(model, val_data)
    test_acc = test(model, test_data)
    print(f'Accuracy before overwriting: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model, wm, wmk, trigger)
    print(f'HMS before overwriting:{hms:.4f}')
    print('------------------------------------')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f},')

    wm_0 = watermark_string_generation(args)
    edge_index = Planetoid(root=args.dataset_path, name='CiteSeer')[0].edge_index.detach().clone().to(device)
    trigger_0 = trigger_generation(model, edge_index, args)
    wmk_0 = watermark_key_generation(model, trigger_0, wm_0, args)
    model_wm = watermark_embedding_2(model, None, val_data, test_data, wm_0, wmk_0, trigger_0, args)
    train_acc = test(model_wm, train_data)
    val_acc = test(model_wm, val_data)
    test_acc = test(model_wm, test_data)
    print(f'Accuracy after overwriting: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model_wm, wm, wmk, trigger)
    print(f'HMS after overwriting:{hms:.4f}')
    print('------------------------------------')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f}\n')



def assess_model_extract(model, model_i, train_data, val_data, test_data, wm, wmk, trigger, args):
    filename = args.results_path + args.dataset + '/setting' + str(args.setting) + '/' + 'model_extract.csv'
    headers =  ['train_acc', 'val_acc', 'test_acc', 'before', 'train_acc', 'val_acc', 'test_acc', 'sage', 'train_acc', 'val_acc', 'test_acc', 'ssg']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    train_acc = test(model, train_data)
    val_acc = test(model, val_data)
    test_acc = test(model, test_data)
    print(f'Accuracy of Model_w: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model, wm, wmk, trigger)
    print(f'HMS of Model_w: {hms:.4f}')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f},')

    subgraph_nodes = torch.zeros(train_data.num_nodes, dtype=torch.bool).to(device)
    subgraph_nodes[torch.randperm(train_data.num_nodes)[:train_data.num_nodes//5 * 4]] = True
    sub_train_data = Data(x=train_data.x[subgraph_nodes], edge_index=subgraph(subgraph_nodes, train_data.edge_index, relabel_nodes=True)[0], y=train_data.y[subgraph_nodes])
    model.eval()
    out = model(sub_train_data.x, sub_train_data.edge_index).softmax(dim=1).detach()

    model_sage = SAGE(sub_train_data.x.shape[1], out.shape[1], [600,400,200]).to(device)
    model_sage = model_extract(model_sage, sub_train_data, out)
    model_ssg = SSG(sub_train_data.x.shape[1], out.shape[1], [600,400,200]).to(device)
    model_ssg = model_extract(model_ssg, sub_train_data, out)

    train_acc = test(model_sage, train_data)
    val_acc = test(model_sage, val_data)
    test_acc = test(model_sage, test_data)
    print(f'Accuracy of the surrogate model extracted SAGE: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model_sage, wm, wmk, trigger)
    print(f'HMS of the surrogate model extracted SAGE: {hms:.4f}')
    print('------------------------------------')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f},')

    train_acc = test(model_ssg, train_data)
    val_acc = test(model_ssg, val_data)
    test_acc = test(model_ssg, test_data)
    print(
        f'Accuracy of the surrogate model extracted SSG: Train: {train_acc:.4f}, Val:{val_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model_ssg, wm, wmk, trigger)
    print(f'HMS of the surrogate model extracted SSG: {hms:.4f}')
    print('------------------------------------')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}, {hms:.4f}\n')


