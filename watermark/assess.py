import copy
import csv
import os.path

import numpy as np
from torch_geometric.utils import subgraph
from torch_geometric.datasets import Planetoid, AttributedGraphDataset
from watermark.watermark import *
from watermark.robust import *
from utils.utils import test
from utils.models import SAGE, SSG


def setting(model_o, model_ind, data, args):
    if args.setting == 1:
        return setting1(copy.deepcopy(model_o), copy.deepcopy(model_ind), data, args)
    elif args.setting == 2:
        return setting2(copy.deepcopy(model_o), copy.deepcopy(model_ind), data, args)
    else:
        raise ValueError('Error: Wrong setting!')


def setting1(model_o, model_ind, data, args):
    filename = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'setting1.csv'
    headers = ['Mo train acc', 'Mo test acc', 'Mi train acc', 'Mi test acc', 'Mi hms', 'Mw train acc', 'Mw test acc', 'Mw hms', 'Time']
    if (not os.path.isfile(filename)) and (model_ind is not None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    new_line = []
    # assess before embedding watermark
    train_acc, test_acc = test(model_o, data, args)
    print(f'Accuracy on the original model: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    new_line.append(train_acc)
    new_line.append(test_acc)

    # embedding watermark
    if args.paradigm == 'transductive':
        trigger = data
    elif args.paradigm == 'inductive':
        trigger = data[0]
    else:
        raise ValueError('Error: Wrong paradigm!')

    start_time = time.time()
    wm = watermark_string_generation(args)
    wmk = watermark_key_generation(model_o, trigger, args)
    model_w = watermark_embedding_1(model_o, data, wm, wmk, trigger, args)
    end_time = time.time()
    sub_time = end_time - start_time

    # acc and bcr of model_ind
    if model_ind is not None:
        print('Independently trained model watermark embedding')
        wm_ind = watermark_string_generation(args)
        trigger_ind = copy.deepcopy(trigger)
        wmk_ind = watermark_key_generation(model_ind, trigger_ind, args)
        model_ind = watermark_embedding_1(model_ind, data, wm_ind, wmk_ind, trigger_ind, args)
        hms = watermark_verification(model_ind, wm, wmk, trigger)
        print(f'HMS of the independently trained model: {hms:.4f}')
        train_acc, test_acc = test(model_ind, data, args)
        new_line.append(train_acc)
        new_line.append(test_acc)
        new_line.append(hms)

    # acc and bcr of model_wm
    hms = watermark_verification(model_w, wm, wmk, trigger)
    train_acc, test_acc = test(model_w, data, args)
    print(f'Accuracy on Model_w: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    print(f'HMS on Model_w: {hms:.4f}')
    print(f'Time of Model_w: {sub_time:.4f}')
    new_line.append(train_acc)
    new_line.append(test_acc)
    new_line.append(hms)
    new_line.append(sub_time)
    print('--------------------------------------------------')
    if model_ind is not None:
        new_line = np.array(new_line)
        with open(filename, mode='a', newline='') as file:
            csv.writer(file).writerow(new_line)
    return model_w, wm, wmk, trigger, model_ind



def setting2(model_o, model_ind, data, args):
    filename = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'setting2.csv'
    headers = ['Mo train acc', 'Mo test acc', 'Mi train acc', 'Mi test acc', 'Mi hms', 'Mw train acc', 'Mw test acc', 'Mw hms', 'Time']
    if not os.path.isfile(filename) and (model_ind is not None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    new_line = []
    # assess before embedding watermark
    train_acc, test_acc = test(model_o, data, args)
    print(f'Accuracy on the original model: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    new_line.append(train_acc)
    new_line.append(test_acc)

    # embedding watermark
    edge_index = Planetoid(root=args.dataset_path, name='CiteSeer', split='public')[0].edge_index.detach().clone().to(args.device)
    trigger = trigger_generation(model_o, edge_index, args)
    start_time = time.time()
    wm = watermark_string_generation(args)
    wmk = watermark_key_generation(model_o, trigger, args)
    model_w = watermark_embedding_2(model_o, data, wm, wmk, trigger, args)
    end_time = time.time()
    sub_time = end_time - start_time

    # acc and bcr of model_ind
    if model_ind is not None:
        print('Independently trained model watermark inject')
        wm_ind = watermark_string_generation(args)
        wmk_ind = watermark_key_generation(model_ind, trigger, args)
        model_ind = watermark_embedding_2(model_ind, data, wm_ind, wmk_ind, trigger, args)
        hms = watermark_verification(model_ind.to(args.device), wm, wmk, trigger)
        print(f'HMS on the independently trained model: {hms:.4f}')
        train_acc, test_acc = test(model_ind, data, args)
        new_line.append(train_acc)
        new_line.append(test_acc)
        new_line.append(hms)

    # acc and bcr of model_wm
    hms = watermark_verification(model_w, wm, wmk, trigger)
    train_acc, test_acc = test(model_w, data, args)
    print(f'Accuracy on the watermarked model: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    print(f'HMS on Model_w: {hms:.4f}')
    print(f'Time of Model_w: {sub_time:.4f}')
    new_line.append(train_acc)
    new_line.append(test_acc)
    new_line.append(hms)
    new_line.append(sub_time)
    print('--------------------------------------------------')
    if model_ind is not None:
        new_line = np.array(new_line)
        with open(filename, mode='a', newline='') as file:
            csv.writer(file).writerow(new_line)
    return model_w, wm, wmk, trigger, model_ind


def assess_pruning(model_w, model_ind, data, wm, wmk, trigger, args):
    filename_w = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'pruning_w.csv'
    headers = [item for i in range(0, 101, 10) for item in [f'{i}%test_acc']] + [item for i in range(0, 101, 10) for item in [f'{i}%hms']]
    if not os.path.isfile(filename_w):
        with open(filename_w, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    tac_list = []
    hms_list = []
    for i in range(11):
        model_copy = model_pruning(copy.deepcopy(model_w), i / 10)
        train_acc, test_acc = test(model_copy, data, args)
        print(f'Accuracy after pruning: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        hms = watermark_verification(model_copy, wm, wmk, trigger)
        print(f'HMS with {i / 10} pruning:{hms:.4f}')
        print('------------------------------------')
        tac_list.append(test_acc)
        hms_list.append(hms)

    tac_list = np.round(np.array(tac_list), 2)
    bcr_list = np.round(np.array(hms_list), 3)
    with open(filename_w, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, bcr_list]))

    filename_i = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'pruning_i.csv'
    if not os.path.isfile(filename_i):
        with open(filename_i, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    tac_list = []
    hms_list = []
    for i in range(11):
        model_copy = model_pruning(copy.deepcopy(model_ind), i / 10)
        train_acc, test_acc = test(model_copy, data, args)
        print(f'Accuracy of Model_ind after pruning of: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        hms = watermark_verification(model_copy, wm, wmk, trigger)
        print(f'HMS of Model_ind with {i / 10} pruning:{hms:.4f}')
        print('------------------------------------')
        tac_list.append(test_acc)
        hms_list.append(hms)

    tac_list = np.round(np.array(tac_list), 2)
    bcr_list = np.round(np.array(hms_list), 3)
    with open(filename_i, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, bcr_list]))



def assess_fine_tuning(model_w, model_ind, data, wm, wmk, trigger, args):
    filename = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'fine_tuning_w.csv'
    headers = [item for i in range(201) for item in [f'{i}test_acc']] + [item for i in range(201) for item in [f'{i}hms']]
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    _, tac_list, hms_list =fine_tuning(copy.deepcopy(model_w), data, wm, wmk, trigger, args, args.lr/2)
    print(f'Accuracy after fine-tuning: Train: {tac_list[-1]:.4f}')
    print(f'HMS after fine-tuning:{hms_list[-1]:.4f}')
    print('------------------------------------')
    tac_list = np.round(np.array(tac_list), 2)
    hms_list = np.round(np.array(hms_list), 3)
    with open(filename, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, hms_list]))

    filename = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'fine_tuning_i.csv'
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    _, tac_list, hms_list = fine_tuning(copy.deepcopy(model_ind), data, wm, wmk, trigger, args, args.lr / 2)
    print(f'Accuracy of Model_ind after fine-tuning: Train: {tac_list[-1]:.4f}')
    print(f'HMS of Model_ind after fine-tuning:{hms_list[-1]:.4f}')
    print('------------------------------------')
    tac_list = np.round(np.array(tac_list), 2)
    hms_list = np.round(np.array(hms_list), 3)
    with open(filename, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, hms_list]))



def assess_overwriting(model, data, wm, wmk, trigger, args):
    model = copy.deepcopy(model)
    filename = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'overwriting.csv'
    headers =  ['original train_acc', 'original test_acc', 'original HMS', 'overwriting train_acc', 'overwriting test_acc', 'overwriting HMS']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    train_acc, test_acc = test(model, data, args)
    print(f'Accuracy before overwriting: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model, wm, wmk, trigger)
    print(f'HMS before overwriting:{hms:.4f}')
    print('------------------------------------')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {test_acc:.4f}, {hms:.4f},')

    wm_0 = watermark_string_generation(args)
    edge_index = AttributedGraphDataset(root=args.dataset_path, name='Wiki')[0].edge_index.detach().clone().to(args.device)
    trigger_0 = trigger_generation(model, edge_index, args)
    wmk_0 = watermark_key_generation(model, trigger_0, args)
    model_wm = watermark_embedding_2(model, data, wm_0, wmk_0, trigger_0, args)
    train_acc, test_acc = test(model_wm, data, args)
    print(f'Accuracy after overwriting: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model_wm, wm, wmk, trigger)
    print(f'HMS after overwriting:{hms:.4f}')
    print('------------------------------------')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {test_acc:.4f}, {hms:.4f}\n')

    del edge_index
    del trigger_0
    del wmk_0
    del model_wm
    torch.cuda.empty_cache()



def assess_unlearning(model_w, data, wm, wmk, trigger, args):
    filename = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'unlearning.csv'
    headers = [item for i in range(201) for item in [f'{i}test_acc']] + [item for i in range(201) for item in [f'{i}hms']]
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    _, tac_list, hms_list = unlearning(copy.deepcopy(model_w), data, wm, wmk, trigger, args, args.lr / 2)
    print(f'Accuracy after unlearning: Train: {tac_list[-1]:.4f}')
    print(f'HMS after unlearning:{hms_list[-1]:.4f}')
    print('------------------------------------')
    tac_list = np.round(np.array(tac_list), 2)
    hms_list = np.round(np.array(hms_list), 3)
    with open(filename, mode='a', newline='') as file:
        csv.writer(file).writerow(np.concatenate([tac_list, hms_list]))



def assess_model_extract(model, data, wm, wmk, trigger, args):
    filename = args.results_path + args.dataset + '/' + args.paradigm + '/setting' + str(args.setting) + '/' + 'model_extract.csv'
    headers =  ['original train_acc', 'original test_acc', 'original HMS', 'SAGE_t train_acc', 'SAGE_t test_acc', 'SAGE_t HMS', 'SAGE_a train_acc', 'SAGE_a test_acc', 'SAGE_a HMS']
        #, 'SSG_t train_acc', 'SSG_t test_acc', 'SSG_t HMS', 'SSG_a train_acc', 'SSG_a test_acc', 'SSG_a HMS']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow(headers)

    train_acc, test_acc = test(model, data, args)
    print(f'Accuracy of Model_w: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model, wm, wmk, trigger)
    print(f'HMS of Model_w: {hms:.4f}')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {test_acc:.4f}, {hms:.4f},')

    # SAGE
    # training graph query
    if args.paradigm == 'transductive':
        out = model(data.x, data.edge_index)[data.train_mask].softmax(dim=1).detach()
        model_sage = SAGE(data.x.shape[1], out.shape[1], args.hidden_channels).to(args.device)
        model_sage = model_extract(model_sage, data, out, args, training_graph=True)
    elif args.paradigm == 'inductive':
        out = model(data[0].x, data[0].edge_index).softmax(dim=1).detach()
        model_sage = SAGE(data[0].x.shape[1], out.shape[1], args.hidden_channels).to(args.device)
        model_sage = model_extract(model_sage, data, out, args, training_graph=True)
    else:
        raise ValueError('Unknown paradigm!')

    train_acc, test_acc = test(model_sage, data, args)
    print(f'Accuracy of the surrogate model extracted SAGE: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model_sage, wm, wmk, trigger)
    print(f'HMS of the surrogate model extracted SAGE: {hms:.4f}')
    print('------------------------------------')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {test_acc:.4f}, {hms:.4f},')


    # Adversary graph query
    if args.paradigm == 'transductive':
        out = model(data.x, data.edge_index)[data.val_mask].softmax(dim=1).detach()
        model_sage = SAGE(data.x.shape[1], out.shape[1], args.hidden_channels).to(args.device)
        model_sage = model_extract(model_sage, data, out, args, training_graph=False)
    elif args.paradigm == 'inductive':
        out = model(data[1].x, data[1].edge_index).softmax(dim=1).detach()
        model_sage = SAGE(data[0].x.shape[1], out.shape[1], args.hidden_channels).to(args.device)
        model_sage = model_extract(model_sage, data, out, args, training_graph=False)
    else:
        raise ValueError('Unknown paradigm!')

    train_acc, test_acc = test(model_sage, data, args)
    print(f'Accuracy of the surrogate model extracted SAGE: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    hms = watermark_verification(model_sage, wm, wmk, trigger)
    print(f'HMS of the surrogate model extracted SAGE: {hms:.4f}')
    print('------------------------------------')
    with open(filename, 'a') as file:
        file.write(f'{train_acc:.4f}, {test_acc:.4f}, {hms:.4f} \n')

    # SSG
    # training graph query
    # if args.paradigm == 'transductive':
    #     out = model(data.x, data.edge_index)[data.train_mask].softmax(dim=1).detach()
    #     model_ssg = SSG(data.x.shape[1], out.shape[1], args.hidden_channels).to(args.device)
    #     model_ssg = model_extract(model_ssg, data, out, args, training_graph=True)
    # elif args.paradigm == 'inductive':
    #     out = model(data[0].x, data[0].edge_index).softmax(dim=1).detach()
    #     model_ssg = SSG(data[0].x.shape[1], out.shape[1], args.hidden_channels).to(args.device)
    #     model_ssg = model_extract(model_ssg, data, out, args, training_graph=True)
    # else:
    #     raise ValueError('Unknown paradigm!')
    #
    # train_acc, test_acc = test(model_ssg, data, args)
    # print(f'Accuracy of the surrogate model extracted SSG: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    # hms = watermark_verification(model_ssg, wm, wmk, trigger)
    # print(f'HMS of the surrogate model extracted SSG: {hms:.4f}')
    # print('------------------------------------')
    # with open(filename, 'a') as file:
    #     file.write(f'{train_acc:.4f}, {test_acc:.4f}, {hms:.4f},')
    #
    # # Adversary graph query
    # if args.paradigm == 'transductive':
    #     out = model(data.x, data.edge_index)[data.val_mask].softmax(dim=1).detach()
    #     model_ssg = SSG(data.x.shape[1], out.shape[1], args.hidden_channels).to(args.device)
    #     model_ssg = model_extract(model_ssg, data, out, args, training_graph=False)
    # elif args.paradigm == 'inductive':
    #     out = model(data[1].x, data[1].edge_index).softmax(dim=1).detach()
    #     model_ssg = SSG(data[0].x.shape[1], out.shape[1], args.hidden_channels).to(args.device)
    #     model_ssg = model_extract(model_ssg, data, out, args, training_graph=False)
    # else:
    #     raise ValueError('Unknown paradigm!')
    #
    # train_acc, test_acc = test(model_ssg, data, args)
    # print(f'Accuracy of the surrogate model extracted SSG: Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    # hms = watermark_verification(model_ssg, wm, wmk, trigger)
    # print(f'HMS of the surrogate model extracted SSG: {hms:.4f}')
    # print('------------------------------------')
    # with open(filename, 'a') as file:
    #     file.write(f'{train_acc:.4f}, {test_acc:.4f}, {hms:.4f} \n')




