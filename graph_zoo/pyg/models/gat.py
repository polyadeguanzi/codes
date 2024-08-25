import torch
from torch import nn
import sys
sys.path.append("/mae_pat/rui.guan/graph_model/")
from pyg.layers.GATConv import MultiHeadGATConv

class GAT(nn.Module):
    """
    ## Graph Attention Network (GAT)

    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(self, in_feats: int, n_hidden: int, out_feats: int, alpha: int, n_heads: int, dropout: float):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        """
        super().__init__()

        self.layer1 = MultiHeadGATConv(in_feats, n_hidden, n_heads, alpha, drop_prob=0.0)
        self.activation = nn.ELU()
        self.output = MultiHeadGATConv(n_hidden, out_feats, n_heads, alpha, drop_prob=0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)