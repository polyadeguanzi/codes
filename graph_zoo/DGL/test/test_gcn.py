
import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.sparse as dglsp
import torch
import sys
#sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo")
sys.path.append("/mae_pat/xi.sun/venus-zoo")
from graph_zoo.DGL.models.gcn import GCN
import nvtx
gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")


if torch.cuda.is_available():
        device = torch.device("cuda:0")
else:
    import torch_gcu
    device = torch_gcu.gcu_device()

net = GCN(in_feats=1433, h_feats=8, num_classes=7).to(device)
###############################################################################
# We load the cora dataset using DGL's built-in data module.

from dgl.data import CoraGraphDataset


def load_cora_data(device):
    dataset = CoraGraphDataset()
    g = dataset[0].to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    return g, features, labels, train_mask, test_mask


###############################################################################
# When a model is trained, we can use the following method to evaluate
# the performance of the model on the test dataset:


def evaluate(model, A, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(A, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


###############################################################################
# We then train the network as follows:

import time

import numpy as np

g, features, labels, train_mask, test_mask = load_cora_data(device)
# Add edges between each node and itself to preserve old node representations
g.add_edges(g.nodes(), g.nodes())
N = g.num_nodes()
indices = torch.stack(g.edges())
A = dglsp.spmatrix(indices, shape=(N, N)).to(device)
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(100):
    with nvtx.annotate("ids al12al1", color="blue"):
        if epoch >= 3:
            t0 = time.time()
        net.train()
        logits = net(A,features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
    
    with nvtx.annotate("ids al12al1", color="yellow"):
        acc = evaluate(net, A, features, labels, test_mask)
    print(
        "Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)
        )
    )

