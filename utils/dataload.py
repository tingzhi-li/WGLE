import torch
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, CitationFull
from torch_geometric.utils import subgraph
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data
from utils.config import device, SEED
# datasets for node classification

def load_data_cora(args):
    dataset = Planetoid(root=args.dataset_path, name='Cora', split='random', num_train_per_class=270, num_val=540)
    data = dataset[0]

    edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
    edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
    val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
    edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
    test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

    return train_graph.to(device), val_graph.to(device), test_graph.to(device), dataset.num_features, dataset.num_classes


def load_data_citeseer(args):
    dataset = Planetoid(root=args.dataset_path, name='CiteSeer', split='random', num_train_per_class=400, num_val=760)
    data = dataset[0]

    edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
    edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
    val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
    edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
    test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

    return train_graph.to(device), val_graph.to(device), test_graph.to(device), dataset.num_features, dataset.num_classes


def load_data_pubmed(args):
    dataset = Planetoid(root=args.dataset_path, name='PubMed', split='random', num_train_per_class=4600, num_val=3942)
    data = dataset[0]

    edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
    edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
    val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
    edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
    test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

    return train_graph.to(device), val_graph.to(device), test_graph.to(device), dataset.num_features, dataset.num_classes


def load_data_dblp(args):
    dataset = CitationFull(root=args.dataset_path, name='DBLP', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
    edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
    val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
    edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
    test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

    return train_graph.to(device), val_graph.to(device), test_graph.to(device), dataset.num_features, dataset.num_classes


def load_data_computers(args):
    dataset = Amazon(root=args.dataset_path, name='Computers', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
    edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
    val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
    edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
    test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

    return train_graph.to(device), val_graph.to(device), test_graph.to(device), dataset.num_features, dataset.num_classes


def load_data_photo(args):
    dataset = Amazon(root=args.dataset_path, name='Photo', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
    edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
    val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
    edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
    test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

    return train_graph.to(device), val_graph.to(device), test_graph.to(device), dataset.num_features, dataset.num_classes


def load_data_cs(args):
    dataset = Coauthor(root=args.dataset_path, name='CS', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
    edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
    val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
    edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
    test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

    return train_graph.to(device), val_graph.to(device), test_graph.to(device), dataset.num_features, dataset.num_classes


def load_data_physics(args):
    dataset = Coauthor(root=args.dataset_path, name='Physics', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
    edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
    val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
    edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
    test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)
    return train_graph.to(device), val_graph.to(device), test_graph.to(device), dataset.num_features, dataset.num_classes


def load_data(args):
    torch.manual_seed(SEED)
    if args.dataset == 'Cora':
        return load_data_cora(args)
    elif args.dataset == 'DBLP':
        return load_data_dblp(args)
    elif args.dataset == 'Photo':
        return load_data_photo(args)
    elif args.dataset == 'Computers':
        return load_data_computers(args)
    elif args.dataset == 'CS':
        return load_data_cs(args)
    elif args.dataset == 'Physics':
        return load_data_physics(args)
    elif args.dataset == 'CiteSeer':
        return load_data_citeseer(args)
    elif args.dataset == 'PubMed':
        return load_data_pubmed(args)
    else:
        print('error')
        return None
