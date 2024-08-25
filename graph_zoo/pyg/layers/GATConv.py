import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops


class GATConv(MessagePassing):
    def __init__(self, in_dim, out_dim, alpha, drop_prob=0.0):
        """
            in_dim(int)：输入节点特征的维度
            out_dim(int)：更新后的节点特征维度
        """
        super().__init__(aggr="add")
        self.drop_prob = drop_prob
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.att_weights = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))   # 后续的计算中会拼接两个节点的特征维度，因此形状为[2*out_dim, 1]
        self.leakrelu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.att_weights)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)    # 确保图中的每个节点都自己指向自己
        # 计算 Wh
        h = self.linear(x)  # [feat_num, out_dim]
        # 启动消息传播
        h_prime = self.propagate(edge_index, x=h) #后续会按edge_index在h的dim=-2挑选边节点
        return h_prime

    def message(self, x_i, x_j, edge_index_i):
        # 计算a(Wh_i || wh_j)
        e = torch.matmul((torch.cat([x_i, x_j], dim=-1)), self.att_weights)
        e = self.leakrelu(e)    # [4,3*2]
        # 以i为中心节点，对i的邻接节点做softmax
        alpha = softmax(e, edge_index_i) # index向量是对e中的dim=1维度做分组，edge_index_i[0]=1,代表e[0]属于类别0，分类完成之后，分别对类别做softmax
        alpha = F.dropout(alpha, self.drop_prob, self.training)
        return x_j * alpha

class MultiHeadGATConv(MessagePassing):
    def __init__(self, in_feats, out_feats, num_heads, alpha, drop_prob=0.0):
        super().__init__(aggr="add")
        self.num_heads = num_heads
        self.head_dim = out_feats
        self.drop_prob = drop_prob

        # Initialize linear transformations for each attention head
        self.linear1 = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.att_weights = nn.Parameter(torch.zeros(size=(2 * out_feats, 1)))
        self.leakrelu = nn.LeakyReLU(alpha)
        self.linear2 = nn.Linear(out_feats * num_heads, out_feats, bias=False)
        nn.init.xavier_uniform_(self.att_weights)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)    # 添加节点自连接

        # Linear transformation for all heads
        h = self.linear1(x) # [node_num, head_num * head_dim]
        h_prime = self.propagate(edge_index, x=h)

        return self.linear2(h_prime)

    def message(self, 
                x_i, # [edge_num, head_num * head_dim]
                x_j, # [edge_num, head_num * head_dim]
                edge_index_i):
        # Calculate attention coefficients for all heads
        x_i = x_i.view(-1, self.num_heads, self.head_dim)   # [edge_num, head_num, head_dim]
        x_j = x_j.view(-1, self.num_heads, self.head_dim)   # [edge_num, head_num, head_dim]
        e = torch.matmul((torch.cat([x_i, x_j], dim=-1)), self.att_weights)  # [edge_num, head_num]
        e = self.leakrelu(e) #
        alpha = softmax(e, edge_index_i)    
        alpha = F.dropout(alpha, self.drop_prob, self.training)

        # Weight the neighbor node features by attention coefficients
        output = x_j * alpha # [edge_num, head_num, head_dim]
        return output.view(x_i.shape[0], -1)    # [edge_num, head_num * head_dim]


# if __name__ == "__main__":
    # conv = GATConv(in_feats=3, out_feats=3, alpha=0.2)
    # x = torch.rand(4, 3)    # 图中有4个节点，每个节点的特征维度为3
    # edge_index = torch.tensor(
    #     [[0, 1, 1, 2, 0, 2, 0, 3], [1, 0, 2, 1, 2, 0, 3, 0]], dtype=torch.long)
    # x = conv(x, edge_index)
    # print(x.shape)


    # conv = MultiHeadGATConv(in_feats=3, out_feats=3, num_heads=2, alpha=0.2)
    # x = torch.rand(4, 3)    # 图中有4个节点，每个节点的特征维度为3
    # edge_index = torch.tensor(
    #     [[0, 1, 1, 2, 0, 2, 0, 3], [1, 0, 2, 1, 2, 0, 3, 0]], dtype=torch.long)
    # x = conv(x, edge_index)
    # print(x.shape)