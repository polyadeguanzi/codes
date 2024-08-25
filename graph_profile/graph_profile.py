import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import time
import numpy as np
import sys
from model_profile.profiler import MyProfiler


def profile_model(model, data, inputs_data,epochs=100,model_name='GAT-PYG'):
    # 创建模型和优化器
    optimizer = torch.optim.Adam(model.parameters(),  lr=1e-3)

    # ******************* 初始化profiler类 *******************
    profiler = MyProfiler(batch_size=data.size()[0], model=model_name, warmup_step=20, time_profile_step=50, op_profile_step=5)
    # *******************************************************

    # 训练循环
    total_step=0
    for epoch in range(epochs):
        # ******************* profile step *******************
        with profiler.step_recorder:
            # ******************* 启动op profile *******************
            profiler.op_profiler.start()
            # ******************* profile 前向传播 *******************
            with  profiler.forward_recorder:
                out = model(*inputs_data)
                #out = model(data.x, data.edge_index)
            # ******************* profile 损失函数 *******************
            with profiler.loss_recorder:
                logp = F.log_softmax(out, 1)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

            # ******************* profile 梯度清零 *******************
            with profiler.zerograd_recorder:
                optimizer.zero_grad()        
            
            # ******************* profile 反向传播 *******************
            with profiler.backward_recorder:
                loss.backward()
            # ******************* profile 梯度更新 *******************
            with profiler.updategrad_recorder:
                optimizer.step()
        # ******************* step *******************
        total_step += 1
        profiler.step()

        #if total_step >= (skip_step + time_profile_step + op_profile_step):
        #    break
    # ******************* stop && save*******************
    profiler.stop()
    profiler.save(model = model, inputs=inputs_data, profile_setting_dict={})


if __name__=='__main__':
    import sys
    from torch_geometric.utils import degree
    sys.path.append("/mae_pat/xi.sun/venus-zoo/graph_zoo")
    from pyg.models.PNA import PNA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载Cora数据集
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    data = data.to(device)
    deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
    scalers = ['identity', 'amplification', 'attenuation']
    model = PNA(in_channels=1433, hidden_channels=8, num_layers=2, out_channels=7,aggregators=['max','std'],deg=deg,scalers=scalers).to(device)
    model_name='PNA-PYG'
    epochs=100
    inputs_data =[data.x, data.edge_index]
    profile_model(model,data,inputs_data,epochs,model_name)