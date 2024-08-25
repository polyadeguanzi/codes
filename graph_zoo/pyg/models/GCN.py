import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional,  Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils._trim_to_layer import TrimToLayer
import sys
# sys.path.append("/mae_pat/haley.hou/venus-zoo")
# sys.path.append("/mae_pat/haley.hou/venus-zoo/graph_zoo")
from graph_zoo.pyg.layers.GCNConv import GCNConv4 as GCNConv
# from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_channel, hidden_channel, out_channel):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_channel, hidden_channel)
        self.conv2 = GCNConv(hidden_channel, out_channel)

    def forward(self,data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
