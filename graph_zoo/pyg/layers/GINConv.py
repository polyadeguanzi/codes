from typing import Callable, Optional, Union
from torch.nn import *
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm


class GINConv_(MessagePassing):

    def __init__(self, channel_list, eps: float = 0., train_eps: bool = False,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        assert len(channel_list)==3
        self.nn = Sequential(Linear(in_channels=channel_list[0], out_channels=channel_list[1], bias=False),
                             LayerNorm(channel_list[1]),
                             Linear(in_channels=channel_list[1], out_channels=channel_list[2], bias=False),
                             LayerNorm(channel_list[2]))
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)

        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

