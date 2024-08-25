import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/mae_pat/haley.hou/venus-zoo/graph_zoo/pyg")
from layers.SAGEConvHgx import SAGEConv

from torch_geometric.nn import GraphSAGE

class GraphSage(nn.Module):
    def __init__(self, input_channel, hidden_channel, output_channel):
        super(GraphSage, self).__init__()
        self.sage1 = SAGEConv(in_channels=input_channel, out_channels=hidden_channel)   # in_channels=1433 out_channels=128
        self.sage2 = SAGEConv(in_channels=hidden_channel, out_channels=output_channel)  # in_channels=128  out_channels=128

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # 该层用于正则化，training: flag表示是否在训练
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim=1) # 该部分一般放在里面，跟F.nll_loss对应，一般用于分类问题，包括GNN的节点分类。