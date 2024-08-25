import os

os.environ["DGLBACKEND"] = "pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F
#from dgl.nn.pytorch import GATConv
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo")
sys.path.append("/mae_pat/xi.sun/venus-zoo")
from graph_zoo.DGL.layers.gat_layer import MultiHeadGATLayer


def edge_attention(self, edges):
    # edge UDF for equation (2)
    z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
    a = self.attn_fc(z2)
    return {"e": F.leaky_relu(a)}



def reduce_func(self, nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox["e"], dim=1)
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
    return {"h": h}



class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


