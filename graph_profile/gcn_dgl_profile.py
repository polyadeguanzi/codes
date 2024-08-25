import networkx as nx

from dgl import DGLGraph
import dgl.sparse as dglsp
from dgl.data import citation_graph as citegrh
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo")
from graph_zoo.DGL.models.gcn import GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_profile.profiler import MyProfiler
import time
import numpy as np
from dgl.data import CoraGraphDataset
def load_cora_data(device):
    dataset = CoraGraphDataset()
    g = dataset[0].to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    return g, features, labels, train_mask, test_mask

def gcn_test(): 
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        import torch_gcu
        device = torch_gcu.gcu_device()
    g, features, labels, train_mask, test_mask = load_cora_data(device)
    g.add_edges(g.nodes(), g.nodes())
    N = g.num_nodes()
    indices = torch.stack(g.edges())
    A = dglsp.spmatrix(indices, shape=(N, N)).to(device)
    # create the model, 2 heads, each head has hidden size 8
    net = GCN(in_feats=1433, h_feats=8, num_classes=7).to(device)
    # create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    skip_step=20
    time_profile_step=50
    op_profile_step=5
    # ******************* 初始化profiler类 *******************
    profiler = MyProfiler(batch_size=features.size()[0], model='GCN', warmup_step=20, time_profile_step=50, op_profile_step=5)
    
    # *******************************************************

    for epoch in range(100):
        # ******************* profile step *******************
        with profiler.step_recorder:
            # ******************* 启动op profile *******************
            profiler.op_profiler.start()
            # ******************* profile 前向传播 *******************
            with  profiler.forward_recorder:
                logits = net(A,features)
            # ******************* profile 损失函数 *******************
            with profiler.loss_recorder:
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp[train_mask], labels[train_mask])
            # ******************* profile 梯度清零 *******************
            with profiler.zerograd_recorder:
                optimizer.zero_grad()
            # ******************* profile 反向传播 *******************
            with profiler.backward_recorder:
                loss.backward()
            # ******************* profile 前向传播 *******************
            with profiler.updategrad_recorder:
                optimizer.step()
        
        # ******************* step *******************
        profiler.step()
        # ********************************************
        if epoch >= (skip_step + time_profile_step + op_profile_step):
                break
    # ******************* stop && save*******************
    profiler.stop()
    profiler.save(model = net, inputs=[A,features], profile_setting_dict={})
    # ***************************************************
 

if __name__ == "__main__":
    gcn_test()