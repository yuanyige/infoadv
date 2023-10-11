import torch
from torch_geometric.nn import GCNConv  
import torch.nn as nn

def select_model(base_model):
    if base_model=='gcn':
        return GCNConv

class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model='gcn', use_ub = True):
        super(Encoder, self).__init__()
        self.base_model = select_model(base_model)
        self.use_ub = use_ub
        self.activation = activation

        self.gc1 = self.base_model(in_channels, 2 * out_channels)
        self.gc2 = self.base_model(2 * out_channels, out_channels)
        if self.use_ub:
            self.gc3 = self.base_model(2 * out_channels, out_channels)
        

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        hidden = self.activation(self.gc1(x, edge_index))
        x1 = self.activation(self.gc2(hidden, edge_index))
        
        if self.use_ub:
            x2 = self.activation(self.gc3(hidden, edge_index))
            return x1, x2
        else:
            return x1



# from torch_sparse import SparseTensor
# model = GCNConv(1,5)
# a=torch.Tensor([[0.9],[0.8],[0.6]]).cuda(0)
# b=SparseTensor.from_dense(torch.Tensor([[0,1,1],[0,0,1],[0,0,0]])).cuda(0)
# print(model(a,b))


