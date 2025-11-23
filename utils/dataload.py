import torch
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, CitationFull, FacebookPagePage, AttributedGraphDataset, Flickr
from torch_geometric.utils import subgraph
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data
from utils.config import SEED
# datasets for node classification



def load_data_cora(args):
    num_train_per_class = int(args.train_val_test[0] * 2708 / 7 )
    num_test = int(args.train_val_test[2] * 2708 )
    dataset = Planetoid(root=args.dataset_path, name='Cora', split='random', num_train_per_class=num_train_per_class, num_test=num_test)
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')



def load_data_citeseer(args):
    num_train_per_class = int(args.train_val_test[0] * 3327 / 6)
    num_test = int(args.train_val_test[2] * 3327)
    dataset = Planetoid(root=args.dataset_path, name='CiteSeer', split='random', num_train_per_class=num_train_per_class,num_test=num_test)
    data = dataset[0]
    args.coe = 2.0

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')



def load_data_pubmed(args):
    num_train_per_class = int(args.train_val_test[0] * 19717 / 3)
    num_test = int(args.train_val_test[2] * 19717)
    dataset = Planetoid(root=args.dataset_path, name='PubMed', split='random', num_train_per_class=num_train_per_class, num_val=num_test)
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')



def load_data_dblp(args):
    dataset = CitationFull(root=args.dataset_path, name='DBLP', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)

        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')



def load_data_computers(args):
    dataset = Amazon(root=args.dataset_path, name='Computers', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)
        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')


def load_data_photo(args):
    dataset = Amazon(root=args.dataset_path, name='Photo', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)
        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')



def load_data_cs(args):
    dataset = Coauthor(root=args.dataset_path, name='CS', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)
        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')



def load_data_physics(args):
    dataset = Coauthor(root=args.dataset_path, name='Physics', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)
        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')



def load_data_facebook(args):
    dataset = FacebookPagePage(root=args.dataset_path, transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)
        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')


def load_data_blog(args):
    dataset = AttributedGraphDataset(root=args.dataset_path, name='BlogCatalog', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask], edge_index=edge_index)
        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')



def load_data_flickr(args):
    dataset = AttributedGraphDataset(root=args.dataset_path, name='Flickr', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
    data = dataset[0]
    data.x = data.x.to_dense().to(torch.float32)

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x.to_dense()[data.train_mask], y=data.y.to_dense()[data.train_mask], edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x.to_dense()[data.val_mask], y=data.y.to_dense()[data.val_mask], edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x.to_dense()[data.test_mask], y=data.y.to_dense()[data.test_mask], edge_index=edge_index)
        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')


def load_data(args):
    torch.manual_seed(SEED)
    if args.dataset == 'Cora':
        data = load_data_cora(args)
    elif args.dataset == 'DBLP':
        data =  load_data_dblp(args)
    elif args.dataset == 'Photo':
        data =  load_data_photo(args)
    elif args.dataset == 'Computers':
        data = load_data_computers(args)
    elif args.dataset == 'CS':
        data =  load_data_cs(args)
    elif args.dataset == 'Physics':
        data = load_data_physics(args)
    elif args.dataset == 'CiteSeer':
        data = load_data_citeseer(args)
    elif args.dataset == 'PubMed':
        data = load_data_pubmed(args)
    elif args.dataset == 'Blog':
        data = load_data_blog(args)
    elif args.dataset == 'Flickr':
        data = load_data_flickr(args)
    else:
        raise ValueError('Error: Unknow dataset!')
    torch.seed()
    return data
