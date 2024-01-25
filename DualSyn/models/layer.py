import os
import sys
import torch
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.typing import Adj
import torch.nn.functional as F
from torch_sparse import SparseTensor
from typing import Union, Tuple, Optional
from torch_geometric.nn import TransformerConv, GCNConv

class DropBlock:
    def __init__(self, dropping_method: str):
        super(DropBlock, self).__init__()
        self.dropping_method = dropping_method

    def drop(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.dropping_method == 'DropNode':  
            x = x * torch.bernoulli(torch.ones(x.size(0), 1) - drop_rate).to(x.device)
            x = x / (1 - drop_rate)  
        elif self.dropping_method == 'DropEdge':
            edge_reserved_size = int(edge_index.size(1) * (1 - drop_rate))
            if isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
                edge_index = torch.stack((row, col))
            perm = torch.randperm(edge_index.size(1))
            edge_index = edge_index.t()[perm][:edge_reserved_size].t()
        elif self.dropping_method == 'Dropout': 
            x = F.dropout(x, drop_rate)

        return x, edge_index


class GNNLayer(torch.nn.Module):
    def __init__(self,
        #in_channels: Union[int, Tuple[int, int]],
        in_channels: int, 
        out_channels: int,
        dropping_method: str,
        heads: int ,
        #transform_first: bool = False,
        ):

        super(GNNLayer, self).__init__()
        
        self.dropping_method = dropping_method
        self.drop_block = DropBlock(dropping_method)
        #self.transform_first = transform_first

        self.backbone = TransformerConv(in_channels, out_channels, heads )
        # self.backbone = GCNConv(in_channels, out_channels)
       

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        message_drop = 0
        if self.dropping_method == 'DropMessage':
            message_drop = drop_rate

        x, edge_index = self.drop_block.drop(x, edge_index, drop_rate)

        # if self.transform_first:
        #     x = x.matmul(self.weight)

        out = self.backbone(x, edge_index)

        # if not self.transform_first:
        #     out = out.matmul(self.weight)
        # if self.bias is not None:
        #     out += self.bias

        return out
