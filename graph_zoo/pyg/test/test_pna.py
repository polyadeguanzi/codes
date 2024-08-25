import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
import time
import numpy as np
import sys
from torch_geometric.utils import degree
sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo")
from pyg.models.PNA import PNA

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]



# 创建模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
scalers = ['identity', 'amplification', 'attenuation']
model = PNA(in_channels=1433, hidden_channels=8, num_layers=2, out_channels=7,aggregators=['max','std'],deg=deg,scalers=scalers).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练循环
dur = []
for epoch in range(100):
    t0 = time.time()
    
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    logp = F.log_softmax(out, 1)
    loss = F.nll_loss(logp[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    dur.append(time.time() - t0)

    print(
        "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur) if dur else 0
        )
    )

# 测试模型
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
print(pred.shape)
assert pred.shape == (2708,)#测试预测结果的形状是否符合预期 ;2708--节点数量
assert pred.isnan().sum() == 0  #测试预测结果数值是否存在nan
assert pred.isinf().sum() == 0  #测试预测结果数值是否存在inf
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
