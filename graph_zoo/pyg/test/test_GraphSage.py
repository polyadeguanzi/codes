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
from torch.utils.data import dataloader
# import pyg_lib
# import torch_sparse

from models.GraphSageHgx import GraphSage
# from torch_geometric.nn import GraphSAGE as GraphSage
def evaluate(model, feature, edge_index, labels, mask):
    # 此处输出的是正确匹配的数量，最后在外面同意除len
    model.eval()
    with torch.no_grad():
        out = model(feature, edge_index)
        out = out[mask]
        _, indices = torch.max(out, dim=1)
        labels = labels[mask]
        correct = torch.sum(indices == labels)
        out = correct.item() * 1.0
    model.train()
    return out

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
    datefmt='%a %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('GraphSage')


# device = torch.device("cpu")
device = torch.device("cuda:7" if torch.cuda.is_available else "cpu")

data = Reddit(root="/mae_pat/haley.hou",transform=T.NormalizeFeatures())._data

# dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]

# dataloader将所有中心节点都取一遍时结束；每个批次的数据加载都是以目标节点为中心向外扩展邻居的；
loader = NeighborLoader(
    data,
    num_neighbors=[10,10], # 每层采样10个邻居
    batch_size=20, # 从大图中选取中心结点的个数（每批次采样的点）
    input_nodes=data.train_mask, # 所需要采样的中心结点集合（全部采样点）
    num_workers=4 # 数据加载线程数
)
    # data: 要求加载 torch_geometric.data.Data 或者 torch_geometric.data.HeteroData 类型数据；
    # num_neighbors: 每轮迭代要采样邻居节点的个数，即第i-1轮要为每个节点采样num_neighbors[i]个节点，如果为-1，则代表所有邻居节点都将被包含(一阶相邻邻居)，在异构图中，还可以使用字典来表示每个单独的边缘类型要采样的邻居数量；
    # input_nodes : 中心节点集合，用来指导采样一个mini-batch内的节点，如果为None，则代表包含data中的所有节点。如果设置为 None，将考虑所有节点。在异构图中，需要作为包含节点类型和节点索引的元组传递。 （默认值：None）
    # input_time (torch.Tensor, optional) – 可选值，用于覆盖 input_nodes 中给定的输入节点的时间戳。如果未设置，将使用 time_attr 中的时间戳作为默认值（如果存在）。需要设置 time_attr 才能使其工作。 （默认值：None）
    # replace (bool, optional) – 如果设置为 True，将进行替换采样。 （默认值：False）
    # directed (bool, optional) – 如果设置为 False，将包括所有采样节点之间的所有边。 （默认值：True）
    # disjoint (bool, optional) – 如果设置为 :obj: True，每个种子节点将创建自己的不相交子图。如果设置为 True，小批量输出将有一个批量向量保存节点到它们各自子图的映射。在时间采样的情况下将自动设置为 True。 （默认值：False） 
    # temporal_strategy (str, optional) -- 使用时间采样时的采样策略（“uniform”、“last”）。如果设置为“uniform”，将在满足时间约束的邻居之间统一采样。如果设置为“last”，将对满足时间约束的最后 num_neighbors 进行采样。 （默认值：“uniform”）
    #     transform (callable, optional) – 一个函数/转换，它接受一个采样的小批量并返回一个转换后的版本。 （默认值：None）
    # transform_sampler_output (callable, optional) – 接受 SamplerOutput 并返回转换后版本的函数/转换。 （默认值：无）
    

net = GraphSage(input_channel=data.x.shape[1], hidden_channel=128, output_channel=128).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
net.train()
count = 0
for epoch in range(20):
    loss_batch = []
    time_spend = []
    acc_list = []
    num_data = 0
    for data_batch in loader:
        # logger.info("data_batch.keys()={}".format(data_batch.keys()))
        data_batch = data_batch.to(device)
        if epoch>3:
            begin = time.time()
        optimizer.zero_grad()
        out = net(data_batch.x, data_batch.edge_index)
        loss = F.nll_loss(out[data_batch.train_mask], data_batch.y[data_batch.train_mask])
        loss.backward()
        optimizer.step()
        if epoch>3:
            end = time.time()
            time_spend.append(end - begin)

        acc = evaluate(net, data_batch.x, data_batch.edge_index,data_batch.y, data_batch.val_mask)
        loss_batch.append(loss)
        acc_list.append(acc)
        num_data += data_batch.x.shape[0]
        count+=1
        if count>=10:
            exit()
        

    if epoch % 5 == 0:
        logger.info("mask.shape={}".format(data_batch.val_mask.shape))
        logger.info("data_batch['x'].shape = {}".format(data_batch.x.shape))
        logger.info("data_batch['edge_index'] = {}".format( data_batch.edge_index.shape))
        
        print("epoch {:05d} | loss {:.4f} | acc {:.4f} | time {:.4f}".format(epoch, sum(loss_batch)/len(loss_batch), sum(acc_list)/len(acc_list), sum(time_spend)/len(time_spend) if len(time_spend)>0 else 0))