import os.path as osp

import torch
from tqdm import tqdm

from torch_geometric.datasets import AmazonBook
#from torch_geometric.nn import LightGCN
import sys
sys.path.append('/mae_pat/xi.sun/venus-zoo/graph_zoo')
from pyg.models.LightgGCN import LightGCN
from torch_geometric.utils import degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Amazon')
dataset = AmazonBook(path)
data = dataset[0]
num_users, num_books = data['user'].num_nodes, data['book'].num_nodes
data = data.to_homogeneous().to(device)

# Use all message passing edges as training labels:
batch_size = 8192
mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, mask]
train_loader = torch.utils.data.DataLoader(
    range(train_edge_label_index.size(1)),
    shuffle=True,
    batch_size=batch_size,
)

model = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=64,
    num_layers=2,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        # Step1 Sample positive and negative labels.
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_books,
                          (index.numel(), ), device=device)
        ], dim=0)

        #Step2 concat得到采样后的邻接矩阵
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()
        #Step3 model.forward返回的得分
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)#chunk分割输出，得到正样本得分和负样本得分
        #assert pred.shape == (features.shape[0], 1) #测试预测结果的形状是否符合预期
        assert pos_rank.isnan().sum() == 0  #测试预测结果数值是否存在nan
        assert neg_rank.isnan().sum() == 0  #测试预测结果数值是否存在nan
        assert pos_rank.isinf().sum() == 0  #测试预测结果数值是否存在inf
        assert neg_rank.isinf().sum() == 0 

        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(k: int):

    '''
    进行模型评估
    测量前 k 个推荐的准确率（precision）和召回率（recall）。
    通过计算每个用户的推荐得分，排除训练集中的边，并比较推荐结果和实际存在的边，最终得出模型的推荐性能指标。
    '''

    ## 获取所有节点的嵌入
    emb = model.get_embedding(data.edge_index)
    #分割user和item的嵌入
    user_emb, book_emb = emb[:num_users], emb[num_users:]

    precision = recall = total_examples = 0
    #批处理
    for start in range(0, num_users, batch_size):
        end = start + batch_size

        # 计算用户和所有物品之间的得分
        logits = user_emb[start:end] @ book_emb.t()

        # 用加mask的方式，“去除”训练集中的边
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # 计算 precision and recall:
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True
        # 计算每个用户的实际物品数
        node_count = degree(data.edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        # 计算每个用户得分最高的前k个物品的索引
        topk_index = logits.topk(k, dim=-1).indices

        # 检查推荐的前k个物品中哪些是实际存在的
        isin_mat = ground_truth.gather(1, topk_index)

        # 计算每个用户的准确率，并累加到总准确率中
        precision += float((isin_mat.sum(dim=-1) / k).sum())

        # 计算每个用户的召回率，并累加到总召回率中
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())

        # 统计实际物品数大于0的用户数
        total_examples += int((node_count > 0).sum())

    # 返回平均准确率和召回率
    return precision / total_examples, recall / total_examples


for epoch in range(1, 10):
    loss = train()  
    precision, recall = test(k=20)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
          f'{precision:.4f}, Recall@20: {recall:.4f}')
