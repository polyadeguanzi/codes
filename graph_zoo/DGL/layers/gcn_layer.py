import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.sparse as dglsp

class GCNConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(GCNConv, self).__init__()
        self.W = nn.Linear(in_size, out_size)

    def forward(self, A, X):
        ########################################################################
        # (HIGHLIGHT) Compute the symmetrically normalized adjacency matrix with
        # Sparse Matrix API
        ########################################################################
        I = dglsp.identity(A.shape,device=A.device)
        A_hat = A + I
        D_hat = dglsp.diag(A_hat.sum(0))
        D_hat_invsqrt = D_hat ** -0.5
        return D_hat_invsqrt @ A_hat @ D_hat_invsqrt @ self.W(X)