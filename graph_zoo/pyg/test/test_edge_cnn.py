import torch
import torch.nn.functional as F
import sys
sys.path.append("/mae_pat/haley.hou/venus-zoo/graph_zoo/pyg")
from models.EdgeCNN import EdgeCNN
sys.path.append("/App/conda/envs/conda_gpu/lib/python3.9/site-packages/")
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import numpy as np
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 同步cuda调用 定位错误



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

# 设备
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
# 3:ncu
# 4:nsys

# 加载Cora数据集
# dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]
# 加载Reddit数据集
data = Reddit(root="/mae_pat/haley.hou",transform=T.NormalizeFeatures())._data

# 模型
type_num = torch.unique(data.y).shape[0]
model = EdgeCNN(input_channel=data.num_features, hidden_channel = type_num,out_channel = type_num).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

train_mask = torch.randn(size=(data.x.shape[0],))
train_mask = train_mask>0.99999
print("All center point {}".format(train_mask.sum()))
print("Before center point {}".format(data.train_mask.sum()))
# dataloader
loader = NeighborLoader(
    data,
    num_neighbors=[30,30], # 每层采样5个邻居
    batch_size=1024, # 从大图中选取中心结点的个数（每批次采样的点）
    input_nodes=data.train_mask, # 所需要采样的中心结点集合（全部采样点）
    num_workers=1 # 数据加载线程数
)

# 训练循环
dur = []
count = 0
flag = False
model.train()
for epoch in range(2):
    if flag:
        break
    for data_batch in loader:
        if flag:
            break
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        out = model(data_batch)
        loss = F.nll_loss(out, data_batch.y)
        loss.backward()
        optimizer.step()
        count+=1
        if count>=5:
            flag = True
        
    if epoch%10==0:
        print(
            "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), np.mean(dur)
            )
        )

# # 测试模型,该部分直接把整个大图放进去跑不通，会被killed
# model.eval()
# _, pred = model(data).max(dim=1)
# correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc = correct / int(data.test_mask.sum())
# print('Accuracy: {:.4f}'.format(acc))