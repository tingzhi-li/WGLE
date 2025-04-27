import time
import copy
import random
import torch
import torch.nn.functional as F
import torch_geometric

from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.data import Data
from sklearn.cluster import DBSCAN
from utils.models import GCNv2
from utils.config import device, MINIMUM
from utils.utils import test


def LDDE(y_hat, x, edge):
    distance_x = F.cosine_similarity(x[edge[0]], x[edge[1]])
    y_tilde = torch.log(F.threshold(y_hat, MINIMUM, MINIMUM))
    y_tilde = y_tilde - torch.mean(y_tilde, dim=1, keepdim=True)
    y_tilde = y_tilde / torch.std(y_tilde, dim=1, keepdim=True)
    distance_y = F.cosine_similarity(y_tilde[edge[0]], y_tilde[edge[1]])
    return distance_y-distance_x



def watermark_key_generation(model_o, trigger, wm, args):
    model_o.eval()
    y_hat = model_o(trigger.x, trigger.edge_index).softmax(dim=1)

    if trigger.y is not None:
        mask = trigger.y[trigger.edge_index[0]] != trigger.y[trigger.edge_index[1]]
    else:
        mask = trigger.edge_attr

    mask = (trigger.edge_index[0] < trigger.edge_index[1]) & mask
    mask_true_index = mask.nonzero(as_tuple=True)[0]
    v = LDDE(y_hat, trigger.x, trigger.edge_index[:, mask_true_index])
    _, topk_index = torch.topk(torch.abs(v), args.n_wm, largest=False)
    m = torch.zeros_like(mask, dtype=torch.bool)
    m[mask_true_index[topk_index]] = True

    return m



def watermark_string_generation(args):
    watermark = torch.randint(0, 2, (args.n_wm,), dtype=torch.uint8).to(device)
    return watermark



def trigger_generation(model_o, edge_index, args):
    model_o.eval()

    num_nodes = edge_index.max() + 1
    if isinstance(model_o, GCNv2):
        num_feat = model_o.fc.in_channels
    else:
        num_feat = model_o.layers[0].in_channels
    x = F.hardtanh(torch.randn((num_nodes, num_feat))).to(device)
    x.requires_grad_(True)
    loss_copy = 1000
    x_copy = None
    optimizer_data = torch.optim.Adam([x], lr=args.trigger_lr)
    mask = edge_index[0] < edge_index[1]

    for epoch in range(args.trigger_epochs):
        y_hat = model_o(F.hardtanh(x), edge_index).softmax(dim=1)
        v = LDDE(y_hat, F.hardtanh(x), edge_index[:, mask])
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
                                               sparse=True, ).to(device)
    loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=args.trigger_lr)
    for epoch in range(args.trigger_epochs//10):
        node2vec.train()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
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
    edge_attr = torch.from_numpy(edge_labels == -1).to(device)
    
    trigger = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
    print(trigger)
    return trigger



def watermark_embedding_1(model_o, train_data, val_data, test_data, wm, wk, trigger, args):
    model_w = copy.deepcopy(model_o)
    optimizer_model = torch.optim.Adam(model_w.parameters(), lr=args.wm_lr, weight_decay=args.weight_decay)
    watermark = wm
    wm = wm.to(torch.float32)
    model_w.eval()

    for epoch in range(args.max_epochs):
        model_w.train()
        optimizer_model.zero_grad()
        trigger_y = model_w(trigger.x, trigger.edge_index).softmax(dim=1)

        y = model_w(train_data.x, train_data.edge_index)
        loss1 = F.cross_entropy(y, train_data.y)
        v = LDDE(trigger_y, trigger.x, trigger.edge_index[:, wk])
        loss2 = F.binary_cross_entropy_with_logits(v.flatten(), wm)
        loss = loss1 + args.coe * loss2
        loss.backward()
        optimizer_model.step()

        if epoch % 20 == 0:
            test_acc = test(model_w, test_data)
            bcr = watermark_verification(model_w, watermark, wk, trigger)
            print(f'The watermarked model is generated. Epoch:{epoch}, Loss1:{loss1:.4f}, Loss2:{loss2:.4f}, Test_acc:{test_acc:.4f}, HMS:{bcr}')
            if (bcr > 0.99) & (epoch >= 100):
                return model_w

    return model_w


def watermark_embedding_2(model_o, train_data, val_data, test_data, wm, wk, trigger, args):
    model_w = copy.deepcopy(model_o)
    watermark = wm
    wm = wm.to(torch.float32)
    model_o.eval()
    if isinstance(model_o, GCNv2):
        num_feat = model_o.fc.in_channels
    else:
        num_feat = model_o.layers[0].in_channels

    # pseudo graph generation
    data_x = F.hardtanh(torch.randn((trigger.num_nodes, num_feat))).to(device)
    data_x.requires_grad_(True)
    data_edge_index = barabasi_albert_graph(len(data_x),1).to(device)
    optimizer_data = torch.optim.Adam([data_x], lr=args.wm_lr)
    optimizer_model = torch.optim.Adam(model_w.parameters(), lr=args.wm_lr, weight_decay=args.weight_decay)
    model_w.eval()

    for epoch in range(args.max_epochs):
        optimizer_model.zero_grad()
        trigger_y = model_w(trigger.x, trigger.edge_index).softmax(dim=1)
        v = LDDE(trigger_y, trigger.x, trigger.edge_index[:, wk])

        out = model_w(F.hardtanh(data_x), data_edge_index)
        label = model_o(F.hardtanh(data_x), data_edge_index).softmax(dim=1)
        loss1 = F.cross_entropy(out, label)
        loss2 = F.binary_cross_entropy_with_logits(v.flatten(), wm)
        loss = loss1 + args.coe * loss2
        loss.backward()
        optimizer_model.step()

        if epoch % 5 == 0:
            optimizer_data.zero_grad()
            out = model_w(F.hardtanh(data_x), data_edge_index)
            label = model_o(F.hardtanh(data_x), data_edge_index).softmax(dim=1)
            loss3 = -F.cross_entropy(out, label)
            loss3.backward()
            optimizer_data.step()

        if epoch % 20 == 0:
            bcr = watermark_verification(model_w, watermark, wk, trigger)
            test_acc = test(model_w, test_data)
            print(f'The watermarked model is generated. Epoch:{epoch}, Loss1:{loss1:.4f}, Loss2:{loss2:.4f}, Test_acc:{test_acc:.4f}, HMS:{bcr}')
            if (bcr > 0.99) & (epoch>=100):
                return model_w

    return model_w



def watermark_verification(model, wm, wk, trigger):
    model.eval()
    trigger_y = model(trigger.x, trigger.edge_index).softmax(dim=1)
    v = LDDE(trigger_y, trigger.x, trigger.edge_index[:, wk]).flatten()
    wme = torch.where(v < 0, 0, 1)
    bcr = int ((wme == wm).sum()) / len(wm)
    return bcr

