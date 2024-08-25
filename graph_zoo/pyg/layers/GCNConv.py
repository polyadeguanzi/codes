import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
#from torch_geometric.nn import glorot, zeros  # 如果你使用 glorot 和 zeros，这里也可以导入


# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
#         self.linear = torch.nn.Linear(in_channels, out_channels)
 
#     def forward(self, x, edge_index):

class GCNConv4(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv4, self).__init__(aggr='add')  # "Add" aggregation.
        self.linear = torch.nn.Linear(in_channels, out_channels)
 
    def forward(self, x, edge_index):
        # 添加自环到边索引
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
         
        # 计算节点的度
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
 
        # 进行线性变换并启动消息传递过程
        x = self.linear(x)
        return self.propagate(edge_index, x=x, norm=norm)
 
    def message(self, x_j, norm):
        # 将特征乘以归一化系数
        return norm.view(-1, 1) * x_j
 
    def update(self, aggr_out):
        # 直接返回聚合后的输出
        return aggr_out
    
