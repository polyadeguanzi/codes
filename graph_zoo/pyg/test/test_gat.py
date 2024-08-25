import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
import time
import numpy as np
import sys

sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo")
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练循环
dur = []
for epoch in range(100):
    if epoch >= 3:
        t0 = time.time()
    
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch >= 3:
        dur.append(time.time() - t0)

    print(
        "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur) if dur else 0
        )
    )

# 测试模型
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
