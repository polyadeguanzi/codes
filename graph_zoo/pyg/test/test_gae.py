import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import degree, negative_sampling
import time
import numpy as np
import sys
sys.path.append("/mae_pat/xi.sun/venus-zoo")
from graph_zoo.pyg.models.GAE import GAE
from graph_zoo.pyg.models.GCN import GCN

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

# 将数据移动到设备上（GPU或CPU）
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 初始化模型
in_channels = dataset.num_node_features  # 输入特征维度
hidden_channels = 64  # 隐藏层维度，可以根据需要调整
out_channels = in_channels  # 输出维度通常与输入维度相同
model = GAE(encoder=GCN(in_channels, hidden_channels, out_channels)).to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练循环
dur = []  # 用于记录每个epoch的时间
for epoch in range(100):
    t0 = time.time()
    
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清除梯度
    
    # 编码节点特征和边索引，得到潜在表示
    z = model.encode(data)
    
    # 使用解码器预测边
    out = model.decode(z, data.edge_index)
    
    # 计算重建损失
    #pos_edge_index = data.edge_index[data.train_mask,data.train_mask]  # 正边索引
    mask = data.train_mask[data.edge_index[0]] & data.train_mask[data.edge_index[1]]
    pos_edge_index = data.edge_index[:, mask]  # 训练边索引
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes)  # 负边索引
    
    # 计算重建损失
    loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
    optimizer.zero_grad()  
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    
    dur.append(time.time() - t0)  # 记录时间

    print(
        "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur) if dur else 0
        )
    )

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不计算梯度
    z=model.encode(data)
    mask = data.test_mask[data.edge_index[0]] & data.test_mask[data.edge_index[1]]
    pos_edge_index = data.edge_index[:, mask] 
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes)
    
    print(model.test(z,pos_edge_index,neg_edge_index))
# assert pred.shape == (2708,)#测试预测结果的形状是否符合预期 ;2708--节点数量
# assert pred.isnan().sum() == 0  #测试预测结果数值是否存在nan
# assert pred.isinf().sum() == 0  #测试预测结果数值是否存在inf