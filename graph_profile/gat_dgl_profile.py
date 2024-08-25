import networkx as nx

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo")

from graph_zoo.DGL.models.gat import GAT
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_profile.profiler import MyProfiler
import time
import numpy as np
def load_cora_data(device):
    data = citegrh.load_cora()
    g = data[0].to(device)
    mask = torch.BoolTensor(g.ndata["train_mask"].cpu()).to(device)
    return g, g.ndata["feat"], g.ndata["label"], mask

def gat_test(): 
    device = torch.device("cuda:0")
    g, features, labels, mask = load_cora_data(device)  
    # create the model, 2 heads, each head has hidden size 8
    net = GAT(g, in_dim=features.size()[1], hidden_dim=8, out_dim=7, num_heads=2).to(device)
    # create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    skip_step=20
    time_profile_step=50
    op_profile_step=5
    # ******************* 初始化profiler类 *******************
    profiler = MyProfiler(batch_size=features.size()[0], model='GAT', warmup_step=20, time_profile_step=50, op_profile_step=5)
    
    # *******************************************************

    for epoch in range(100):
        # ******************* profile step *******************
        with profiler.step_recorder:
            # ******************* 启动op profile *******************
            profiler.op_profiler.start()
            # ******************* profile 前向传播 *******************
            with  profiler.forward_recorder:
                logits = net(features)
            # ******************* profile 损失函数 *******************
            with profiler.loss_recorder:
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp[mask], labels[mask])
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
    profiler.save(model = net, inputs=features, profile_setting_dict={})
    # ***************************************************
 

if __name__ == "__main__":
    gat_test()