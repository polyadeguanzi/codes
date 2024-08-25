import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
import time
import numpy as np
import sys

sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo")
from model_profile.profiler import MyProfiler
from pyg.models.gat import GAT

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]


# 创建模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(in_feats=dataset.num_features, 
        n_hidden = 8,
        out_feats = 7, # 节点特征的输出维度
        alpha = 0.2,
        n_heads = 2, 
        dropout = 0.0).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(),  lr=1e-3)

# ******************* 初始化profiler类 *******************
profiler = MyProfiler(batch_size=data.size()[0], model='PYG', warmup_step=20, time_profile_step=50, op_profile_step=5)
# *******************************************************

# 训练循环
dur = []
for epoch in range(100):
    # ******************* profile step *******************
    with profiler.step_recorder:
        # ******************* 启动op profile *******************
        profiler.op_profiler.start()
        if epoch >= 3:
            t0 = time.time()
        
        model.train()
        # ******************* profile 梯度清零 *******************
        with profiler.zerograd_recorder:
            optimizer.zero_grad()
        # ******************* profile 前向传播 *******************
        with  profiler.forward_recorder:
            out = model(data.x, data.edge_index)
        # ******************* profile 损失函数 *******************
        with profiler.loss_recorder:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # ******************* profile 反向传播 *******************
        with profiler.backward_recorder:
            loss.backward()
        # ******************* profile 前向传播 *******************
        with profiler.updategrad_recorder:
            optimizer.step()
    # ******************* step *******************
    profiler.step()
    # ********************************************
    if epoch >= 3:
        dur.append(time.time() - t0)

    print(
        "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur) if dur else 0)
        )
# ******************* stop && save*******************
profiler.stop()
profiler.save(model = model, inputs=[data.x, data.edge_index], profile_setting_dict={})