import torch.nn.functional as F

#train and test
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    y = model(data.x, data.edge_index)
    loss = F.cross_entropy(y, data.y)
    loss.backward()
    optimizer.step()
    return loss


def test(model, data):
    model.eval()
    y = model(data.x, data.edge_index)
    y = y.argmax(dim=1)
    acc = int((y == data.y).sum()) / len(data.y)
    return acc * 100


