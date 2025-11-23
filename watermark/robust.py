import torch
import torch.nn.utils.prune as prune
from utils.utils import test
from utils.models import *
from watermark.watermark import watermark_verification, LDDE


# free-data attack

def model_pruning(model, ratio=0.1):
    for name, param in model.named_modules():
        if isinstance(param, Linear):
            prune.l1_unstructured(param, 'weight', ratio)
            if param.bias is not None:
                prune.l1_unstructured(param, 'bias', ratio)
        if isinstance(param, GCN2Conv):
            prune.l1_unstructured(param, 'weight1', ratio)
        if isinstance(param, ARMAConv):
            prune.l1_unstructured(param, 'bias', ratio)
            prune.l1_unstructured(param, 'init_weight', ratio)
            prune.l1_unstructured(param, 'root_weight', ratio)
            prune.l1_unstructured(param, 'weight', ratio)
    return model


# attack needed dataset

def fine_tuning(model, data, wm, wmk, trigger, args, lr=5e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=1e-5)
    tac_list = []
    bcr_list = []
    train_acc, test_acc = test(model, data, args)
    bcr = watermark_verification(model, wm, wmk, trigger)
    tac_list.append(test_acc)
    bcr_list.append(bcr)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        if args.paradigm == 'transductive':
            y = model(data.x, data.edge_index)
            loss = F.cross_entropy(y[data.val_mask], data.y[data.val_mask])
        elif args.paradigm == 'inductive':
            y = model(data[1].x, data[1].edge_index)
            loss = F.cross_entropy(y, data[1].y)
        else:
            raise ValueError('Wrong paradigm!')

        loss.backward()
        optimizer.step()
        scheduler.step()
        train_acc, test_acc = test(model, data, args)
        bcr = watermark_verification(model, wm, wmk, trigger)
        tac_list.append(test_acc)
        bcr_list.append(bcr)
        if epoch % 20 == 0:
            print(f"Fine-tuning Epoch: {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, HMS: {bcr:.4f}")
    return model, tac_list, bcr_list


def unlearning(model, data, wm, wmk, trigger, args, lr=5e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=1e-5)
    tac_list = []
    bcr_list = []
    train_acc, test_acc = test(model, data, args)
    bcr = watermark_verification(model, wm, wmk, trigger)
    tac_list.append(test_acc)
    bcr_list.append(bcr)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        if args.paradigm == 'transductive':
            y = model(data.x, data.edge_index)
            loss1 = F.cross_entropy(y[data.val_mask], data.y[data.val_mask])
            loss2 = torch.mean(torch.abs(LDDE(F.softmax(y, dim=1), data.x, data.edge_index)))
            loss = loss1 + loss2
        elif args.paradigm == 'inductive':
            y = model(data[1].x, data[1].edge_index)
            loss1 = F.cross_entropy(y, data[1].y)
            loss2 = torch.mean(torch.abs(LDDE(F.softmax(y, dim=1), data[1].x, data[1].edge_index)))
            loss = loss1 + loss2
        else:
            raise ValueError('Error: Wrong paradigm!')

        loss.backward()
        optimizer.step()
        scheduler.step()
        train_acc, test_acc = test(model, data, args)
        bcr = watermark_verification(model, wm, wmk, trigger)
        tac_list.append(test_acc)
        bcr_list.append(bcr)
        if epoch % 20 == 0:
            print(f"Unlearning Loss: {loss.item():.4f}, Train Acc: {train_acc}, Test Acc: {test_acc}, HMS: {bcr:.4f} ")
    return model, tac_list, bcr_list



def model_extract(model, data, out, args, training_graph=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=1e-5)
    for epoch in range(1501):
        optimizer.zero_grad()

        if training_graph:
            if args.paradigm == 'transductive':
                y = model(data.x, data.edge_index)[data.train_mask]
                loss = F.kl_div(F.log_softmax(y, dim=1), out, reduction='sum')
            elif args.paradigm == 'inductive':
                y = model(data[0].x, data[0].edge_index)
                loss = F.kl_div(F.log_softmax(y, dim=1), out, reduction='sum')
            else:
                raise ValueError('Error: wrong paradigm!')
        else:
            if args.paradigm == 'transductive':
                y = model(data.x, data.edge_index)[data.val_mask]
                loss = F.kl_div(F.log_softmax(y, dim=1), out, reduction='sum')
            elif args.paradigm == 'inductive':
                y = model(data[1].x, data[1].edge_index)
                loss = F.kl_div(F.log_softmax(y, dim=1), out, reduction='sum')
            else:
                raise ValueError('Error: wrong paradigm!')

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 300 == 0:
            train_acc, test_acc = test(model, data, args)
            print(f'Model extraction attack is ongoing: Epoch: {epoch:03d}, Loss:{loss:.4f}, Train accuracy:{train_acc:.4f}, Test accuracy:{test_acc:.4f}')
            if epoch < 1001:
                model.train()
            else:
                model.eval()
    return model


