# from torch_geometric.nn.models import EdgeCNN
# from torch_geometric.nn.conv import EdgeConv
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/mae_pat/haley.hou/venus-zoo/graph_zoo/pyg")
from layers.EdgeConv import EdgeConv

class EdgeCNN(nn.Module):
    def __init__(self, input_channel, hidden_channel, out_channel) -> None:
        super(EdgeCNN, self).__init__()
        self.conv1 = EdgeConv(input_channel, hidden_channel)
        self.conv2 = EdgeConv(hidden_channel, out_channel)
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        
