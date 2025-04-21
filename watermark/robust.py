import torch
import torch.nn.utils.prune as prune
from utils.utils import test
from utils.models import *
from watermark.watermark import watermark_verification


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

def fine_tuning(model, data, test_data, wm, wmk, trigger, lr=5e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tac_list = []
    bcr_list = []
    test_acc = test(model, test_data)
    bcr = watermark_verification(model, wm, wmk, trigger)
    tac_list.append(test_acc)
    bcr_list.append(bcr)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        y = model(data.x, data.edge_index)
        loss = F.cross_entropy(y, data.y)
        loss.backward()
        optimizer.step()
        test_acc = test(model, test_data)
        bcr = watermark_verification(model, wm, wmk, trigger)
        tac_list.append(test_acc)
        bcr_list.append(bcr)
        print("loss:", loss.item()) if epoch % 10 == 0 else None
    return model, tac_list, bcr_list



def model_extract(model, data, out):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1501):
        optimizer.zero_grad()
        y = model(data.x, data.edge_index)
        loss = F.kl_div(F.log_softmax(y,dim=1), out, reduction='sum')
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            acc = test(model, data)
            print(f'Model extraction attack is ongoing: Epoch: {epoch:03d}, Loss:{loss:.4f}, Accuracy:{acc:.4f}')
            if epoch < 1000:
                model.train()
            else:
                model.eval()
    return model


