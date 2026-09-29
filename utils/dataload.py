import torch
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, CitationFull, FacebookPagePage, AttributedGraphDataset, Flickr
from torch_geometric.utils import subgraph
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data
from utils.config import SEED
# datasets for node classification



def load_data_cora(args):
    dataset = CitationFull(root=args.dataset_path, name='Cora_ML', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
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
    dataset = CitationFull(root=args.dataset_path, name='CiteSeer', transform=RandomNodeSplit(num_val=args.train_val_test[1], num_test=args.train_val_test[2]))
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


def load_data_OGB_Arxiv(args):
    import numpy as np
    import os
    from ogb.nodeproppred import PygNodePropPredDataset

    try:
        # ── Attempt full dataset load ─────────────────────────────────
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.dataset_path)
        data = dataset[0]
        split_idx = dataset.get_idx_split()

        # Convert OGB split indices to boolean masks (matching PyG convention)
        num_nodes = data.num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']]   = True
        data.test_mask[split_idx['test']]   = True

        # Flatten labels from [N, 1] to [N]
        data.y = data.y.view(-1)

    except (MemoryError, RuntimeError) as e:
        # ── Fallback: load a representative subset ────────────────────
        print(f'Full ogbn-arxiv load failed ({e}).  Loading representative subset ...')

        dataset   = PygNodePropPredDataset(name='ogbn-arxiv', root=args.dataset_path)
        split_idx = dataset.get_idx_split()
        raw_dir   = dataset.raw_dir

        # Load raw arrays from disk (avoids the full tensor conversion)
        x_full = torch.from_numpy(np.load(os.path.join(raw_dir, 'node_feat.npy'))).float()
        y_full = torch.from_numpy(np.load(os.path.join(raw_dir, 'node_label.npy'))).long()
        ei_full = torch.from_numpy(
            np.load(os.path.join(raw_dir, 'edge_index.npy'))
        ).long().t().contiguous()

        train_idx = split_idx['train']
        val_idx   = split_idx['valid']
        test_idx  = split_idx['test']

        # Build a representative subset: all train + val + a random sample of test
        n_test_sample = max(len(train_idx), len(test_idx) // 3)
        test_subset   = test_idx[torch.randperm(len(test_idx))[:n_test_sample]]

        all_idx = torch.cat([train_idx, val_idx, test_subset]).unique().sort()[0]
        idx_map = {int(old): new for new, old in enumerate(all_idx.tolist())}

        ei_sub, _ = subgraph(all_idx, ei_full, relabel_nodes=True)

        data = Data(
            x=x_full[all_idx],
            y=y_full[all_idx].view(-1),
            edge_index=ei_sub,
        )

        def _mask(indices):
            m = torch.zeros(data.num_nodes, dtype=torch.bool)
            mapped = torch.tensor(
                [idx_map[int(i)] for i in indices.tolist() if int(i) in idx_map]
            )
            m[mapped] = True
            return m

        data.train_mask = _mask(train_idx)
        data.val_mask   = _mask(val_idx)
        data.test_mask  = _mask(test_subset)

    # ── Return based on paradigm ──────────────────────────────────────
    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask],
                           edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask],
                         edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask],
                          edge_index=edge_index)
        return train_graph.to(args.device), val_graph.to(args.device), test_graph.to(args.device)
    else:
        raise ValueError('Error: Wrong paradigm!')


def load_data_Roman_empire(args):
    from torch_geometric.datasets import HeterophilousGraphDataset

    dataset = HeterophilousGraphDataset(root=args.dataset_path, name='Roman-empire')
    data = dataset[0]

    # HeterophilousGraphDataset provides 10 random splits as masks of shape
    # [num_nodes, 10].  Use the first split (index 0) consistently.
    data.train_mask = data.train_mask[:, 0]
    data.val_mask   = data.val_mask[:, 0]
    data.test_mask  = data.test_mask[:, 0]

    if args.paradigm == 'transductive':
        return data.to(args.device)
    elif args.paradigm == 'inductive':
        edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
        train_graph = Data(x=data.x[data.train_mask], y=data.y[data.train_mask],
                           edge_index=edge_index)
        edge_index = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0]
        val_graph = Data(x=data.x[data.val_mask], y=data.y[data.val_mask],
                         edge_index=edge_index)
        edge_index = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0]
        test_graph = Data(x=data.x[data.test_mask], y=data.y[data.test_mask],
                          edge_index=edge_index)
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
    elif args.dataset == 'Blog':
        data = load_data_blog(args)
    else:
        raise ValueError('Error: Unknow dataset!')

    # ── Dataset statistics ─────────────────────────────────────────────
    _print_dataset_stats(args.dataset, data, args.paradigm)

    torch.seed()
    return data


def _print_dataset_stats(dataset_name, data, paradigm):
    """Compute and display graph size, edge counts, and node density."""
    def _stats_for_graph(graph, label):
        num_nodes = graph.num_nodes
        # PyG stores undirected graphs with both (i,j) and (j,i);
        # actual undirected edge count is half the stored entries.
        num_edges_directed = graph.edge_index.size(1)
        num_edges = num_edges_directed // 2
        # Density: actual undirected edges / possible undirected edges
        if num_nodes > 1:
            possible_edges = num_nodes * (num_nodes - 1) / 2
            density = num_edges / possible_edges
        else:
            density = 0.0
        print(f'  [{dataset_name}] {label}: '
              f'Nodes={num_nodes},  Edges={num_edges_directed} (directed entries),  '
              f'UndirectedEdges={num_edges},  Density={density:.6f}')

    if paradigm == 'transductive':
        _stats_for_graph(data, 'transductive')
    elif paradigm == 'inductive':
        _stats_for_graph(data[0], 'train')
        _stats_for_graph(data[1], 'val')
        _stats_for_graph(data[2], 'test')
    else:
        raise ValueError('Error: Wrong paradigm!')
