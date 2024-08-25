from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch import Tensor
import torch

class EdgeConv(MessagePassing):
    def __init__(self, in_channel, out_channel, aggr: str = 'max') -> None:
        super(EdgeConv, self).__init__(aggr=aggr)
        self.nn = MLP([2*in_channel, out_channel, out_channel])

    def forward(self, x, edge_index):
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
    
