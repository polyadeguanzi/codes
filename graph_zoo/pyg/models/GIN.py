from torch_geometric.nn import GIN
import torch
from typing import Optional
from torch import Tensor
from torch.nn import *
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import sys
sys.path.append("/mae_pat/haley.hou/venus-zoo/graph_zoo/pyg")
from layers.GINConv import GINConv_


class GIN_(torch.nn.Module):

    def __init__(self,in_channels: int,hidden_channels: int,out_channels:int = None):
        super().__init__()

        self.conv1 = self.init_conv(in_channels, hidden_channels)
        self.conv2 = self.init_conv(hidden_channels, out_channels)

    def init_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        channel_list = [in_channels, out_channels, out_channels]
        return GINConv_(channel_list)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1) # 因为输出之后的损失函数对此有要求


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}')

# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# # 加载Cora数据集
# dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]

# # 创建模型和优化器
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GIN_(in_channels=dataset.num_features, hidden_channels = 8,out_channels = 7).to(device)