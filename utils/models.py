import torch.nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, TransformerConv, SSGConv, GENConv, GCN2Conv, ARMAConv, GINConv, MLP, GATConv, Linear

class GCNv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.fc = Linear(in_channels, hidden_channels[0])
        self.layers = torch.nn.ModuleList()
        for i in range(len(hidden_channels)):
            self.layers.append(GCN2Conv(hidden_channels[0], i/len(hidden_channels)))
        self.out = Linear(hidden_channels[0], out_channels)

    def forward(self, x, edge_index):
        x = x0 = self.fc(x)
        for layer in self.layers:
            x = layer(x, x0, edge_index)
            x = F.elu(x)
        x = self.out(x)
        return x


class SSG(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(SSGConv(in_channels, hidden_channels[0], 0.8))
        for i in range(len(hidden_channels) - 1):
            self.layers.append(SSGConv(hidden_channels[i], hidden_channels[i + 1], 0.8))
        self.out = SSGConv(hidden_channels[-1], out_channels, 0.8)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
        x = self.out(x, edge_index)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels)-1):
            self.layers.append(SAGEConv(hidden_channels[i], hidden_channels[i+1]))
        self.out = SAGEConv(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
        x = self.out(x, edge_index)
        return x


class ARMA(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(ARMAConv(in_channels, hidden_channels[0], act=torch.nn.ELU()))
        for i in range(len(hidden_channels) - 1):
            self.layers.append(ARMAConv(hidden_channels[i], hidden_channels[i + 1], act=torch.nn.ELU()))
        self.out = ARMAConv(hidden_channels[-1], out_channels, act=None)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.out(x, edge_index)
        return x


class GEN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GENConv(in_channels, hidden_channels[0], norm='batch', aggr='mean'))
        for i in range(len(hidden_channels) - 1):
            self.layers.append(GENConv(hidden_channels[i], hidden_channels[i + 1], norm='batch', aggr='mean'))
        self.out = GENConv(hidden_channels[-1], out_channels, norm='batch', aggr='mean')

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
        x = self.out(x, edge_index)
        return x


class GTF(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(TransformerConv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.layers.append(TransformerConv(hidden_channels[i], hidden_channels[i + 1]))
        self.out = TransformerConv(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
        x = self.out(x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GINConv(MLP([in_channels, hidden_channels[0]])))
        for i in range(len(hidden_channels) - 1):
            self.layers.append(GINConv(MLP([hidden_channels[i], hidden_channels[i + 1]])))
        self.out = GINConv(MLP([hidden_channels[-1], out_channels]))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
        x = self.out(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.layers.append(GATConv(hidden_channels[i], hidden_channels[i + 1]))
        self.out = GATConv(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
        x = self.out(x, edge_index)
        return x


def load_model(in_channels, out_channels, args):
    if args.model == 'GCNv2':
        return GCNv2(in_channels, out_channels, args.hidden_channels).to(args.device)
    elif args.model == 'SSG':
        return SSG(in_channels, out_channels, args.hidden_channels).to(args.device)
    elif args.model == 'SAGE':
        return SAGE(in_channels, out_channels, args.hidden_channels).to(args.device)
    elif args.model == 'ARMA':
        return ARMA(in_channels, out_channels, args.hidden_channels).to(args.device)
    elif args.model == 'GTF':
        return GTF(in_channels, out_channels, args.hidden_channels).to(args.device)
    elif args.model == 'GEN':
        return GEN(in_channels, out_channels, args.hidden_channels).to(args.device)
    elif args.model == 'GIN':
        return GIN(in_channels, out_channels, args.hidden_channels).to(args.device)
    elif args.model == 'GAT':
        return GAT(in_channels, out_channels, args.hidden_channels).to(args.device)
    else:
        raise ValueError('Error: Unknown model!')

