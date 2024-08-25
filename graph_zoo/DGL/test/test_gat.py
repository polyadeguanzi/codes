import networkx as nx

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo")
sys.path.append("/mae_pat/xi.sun/venus-zoo")
from graph_zoo.DGL.models.gat import GAT
import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
        device = torch.device("cuda:0")
else:
    import torch_gcu
    device = torch_gcu.gcu_device()

def load_cora_data(device):
    data = citegrh.load_cora()
    g = data[0].to(device)
    train_mask = torch.BoolTensor(g.ndata["train_mask"].cpu()).to(device)
    test_mask = torch.BoolTensor(g.ndata["test_mask"].cpu()).to(device)  # Assuming you have a test mask
    return g, g.ndata["feat"], g.ndata["label"], train_mask, test_mask



##############################################################################
# The training loop is exactly the same as in the GCN tutorial.

import time

import numpy as np
import nvtx

g, features, labels, train_mask, test_mask = load_cora_data(device)


# create the model, 2 heads, each head has hidden size 8
net = GAT(g, in_dim=features.size()[1], hidden_dim=8, out_dim=7, num_heads=2).to(device)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# main loop
dur = []
for epoch in range(100):
    if epoch >= 3:
        t0 = time.time()
    with nvtx.annotate("training", color="blue"):
        logits = net(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask]) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), np.mean(dur)
            )
        )

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        pred = torch.argmax(logp, dim=1)
        correct = torch.sum(pred[mask] == labels[mask]).item()
        accuracy = correct / mask.sum().item()
    return accuracy
with nvtx.annotate("infer", color="yellow"):
# Evaluate the model on the test set
    test_accuracy = evaluate(net, features, labels, test_mask)
print("Test Accuracy: {:.4f}".format(test_accuracy))