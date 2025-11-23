import torch.nn.functional as F

#train and test
def train(model, data, optimizer, args):
    model.train()
    optimizer.zero_grad()

    if args.paradigm == 'transductive':
        y = model(data.x, data.edge_index)
        loss = F.cross_entropy(y[data.train_mask], data.y[data.train_mask])
    elif args.paradigm == 'inductive':
        y = model(data[0].x, data[0].edge_index)
        loss = F.cross_entropy(y, data[0].y)
    else:
        raise ValueError('Error: Wrong paradigm!')

    loss.backward()
    optimizer.step()
    return loss


def test(model, data, args):
    model.eval()

    if args.paradigm == 'transductive':
        y = model(data.x, data.edge_index)
        y = y.argmax(dim=1)
        train_acc = int((y[data.train_mask] == data.y[data.train_mask]).sum()) / len(data.y[data.train_mask])
        test_acc = int((y[data.test_mask] == data.y[data.test_mask]).sum()) / len(data.y[data.test_mask])
    elif args.paradigm == 'inductive':
        y = model(data[0].x, data[0].edge_index)
        y = y.argmax(dim=1)
        train_acc = int((y == data[0].y).sum()) / len(data[0].y)
        y = model(data[2].x, data[2].edge_index)
        y = y.argmax(dim=1)
        test_acc = int((y == data[2].y).sum()) / len(data[2].y)
    else:
        raise ValueError('Error: Wrong paradigm!')

    return train_acc * 100, test_acc * 100


