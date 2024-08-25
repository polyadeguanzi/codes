import logging
import sys
sys.path.append("/mae_pat/haley.hou/venus-zoo/graph_zoo/pyg")
sys.path.append("/App/conda/envs/conda_gpu/lib/python3.9/site-packages")
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import torch
import time

# import pyg_lib
# import torch_sparse

from models.GraphSageHgx import GraphSage
# from torch_geometric.nn import GraphSAGE as GraphSage
from graph_zoo.pyg.models.graph_unet import GraphUNet

def load_data(device):
    #加载数据集
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    g = dataset[0].to(device)

    features = g.x
    edge_idx = g.edge_index
    labels = g.y
    train_mask = g.train_mask
    val_mask = g.val_mask
    test_mask = g.val_mask
    return g, features, edge_idx, labels, train_mask, val_mask, test_mask

def evaluate(model, g, labels, mask):
    model.eval()
    with torch.no_grad():#不追踪梯度
        logits = model(g)#输入data
        logits = logits[mask]#仅指定mask部分进行评估(验证集)
        _, indices = torch.max(logits, dim=1)#对每个样本的预测输出，找到最大值所在的索引，这个索引即为模型的预测类别。
        labels = labels[mask]#真实标签
        correct = torch.sum(indices == labels)
        out = correct.item() * 1.0 / len(labels)#准确率
    model.train()
    return out



# 加载Cora数据集
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g, features, edge_idx, labels, train_mask, val_mask, test_mask = load_data(device)

# 创建模型和优化器
model = GCN(input_channel=features.shape[-1],
            hidden_channel=8,
            out_channel=7).to(device)

model = GraphUNetModel(input_channel=features.shape[-1],
                       hidden_channel=8,
                       out_channel=7,
                       depth=3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 训练循环
dur = []
model.train()
for epoch in range(300):
    optimizer.zero_grad()
    #前几次不准，warm
    if epoch >3 :
        t0 = time.time()
    
    logits = model(g)
    loss = F.nll_loss(logits[train_mask], labels[train_mask])

    loss.backward()
    optimizer.step()#调整模型参数
    
    if epoch >3:
        dur.append(time.time() - t0)

    #在每10个（epoch）之后进行一次评估evaluate，并输出
    if epoch%10==0:
        acc = evaluate(model=model, g=g,labels=labels, mask=val_mask)
        print(
            "Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur) if dur else 0
            )
        )

#最后再输出test的评估结果 测试模型
model.eval()
_, pred = model(g).max(dim=1)
correct = int(pred[test_mask].eq(labels[test_mask]).sum().item())
acc = correct / int(test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))