import torch
import torch.nn.functional as F

from graph_zoo.pyg.models.RENET import RENet
import torch_geometric.datasets 
# 加载 Wiki 数据集
dataset = torch_geometric.datasets.ICEWS18(root='/tmp/ICEWISE18')

print(1)



# 创建模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = dataset.to(device)
model = RENet(num_nodes=23033,
        num_rels=256,
        hidden_channels=256,
        seq_len=10,
        num_layers=2).to(device)
ts=model.pre_transform(10)
data=ts(dataset)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
dur = []
for epoch in range(100):
    
    model.train()
    optimizer.zero_grad()
    out = model(data)
    logp = F.log_softmax(out, 1)
    loss = F.nll_loss(logp[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    

    print(
        "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item()
        )
    )

# 测试模型
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
print(pred.shape)
assert pred.shape == (2708,)#测试预测结果的形状是否符合预期 ;2708--节点数量
assert pred.isnan().sum() == 0  #测试预测结果数值是否存在nan
assert pred.isinf().sum() == 0  #测试预测结果数值是否存在inf
# correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc = correct / int(data.test_mask.sum())
# print('Accuracy: {:.4f}'.format(acc))
