
import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo")
sys.path.append("/mae_pat/xi.sun/venus-zoo")
from graph_zoo.DGL.layers.gcn_layer import GCNConv

# Create a GCN with the GCN layer.
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def forward(self, A, X):
        X = self.conv1(A, X)
        X = F.relu(X)
        return self.conv2(A, X)