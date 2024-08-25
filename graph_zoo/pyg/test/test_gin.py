import torch
import torch.nn.functional as F
import sys
sys.path.append("/mae_pat/haley.hou/venus-zoo/graph_zoo/pyg")
sys.path.append("/App/conda/envs/conda_gpu/lib/python3.9/site-packages/")
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import time
import numpy as np

from models.GIN import GIN_

def evaluate(model, data, labels, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        out = out[mask]
        _, indices = torch.max(out, dim=1)
        labels = labels[mask]
        correct = torch.sum(indices == labels)
        out = correct.item() * 1.0 / len(labels)
    model.train()
    return out


# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

# 创建模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN_(in_channels=dataset.num_features, hidden_channels = 8,out_channels = 7).to(device)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练循环
dur = []
for epoch in range(1000):
    if epoch >= 3:
        t0 = time.time()
    
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch >= 3:
        dur.append(time.time() - t0)

    if epoch%10==0:
        acc = evaluate(model=model, data=data, labels=data.y, mask=data.val_mask)
        print(
            "Epoch {:05d} | Loss {:.4f} |Test acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(),acc, np.mean(dur)
            )
        )

# 测试模型
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))