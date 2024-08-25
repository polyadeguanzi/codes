import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
if not torch.cuda.is_available():
    import torch_gcu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rec_zoo.models.base  import BaseModel, Config, build_component
from rec_zoo.models.registers.register import venus_model
from pyg.layers.GCNConv import GCNConv4 as GCNConv
from pyg.layers.topk_pool import TopKPooling


from torch_geometric.nn import GCNConv
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat

    r"""
    参数:
        in_channels (int): 每个输入样本的大小。
        hidden_channels (int): 每个隐藏样本的大小。
        out_channels (int): 每个输出样本的大小。
        depth (int): U-Net 结构的深度。需要大于等于1
        pool_ratios (float 或 [float], 可选): 每个深度的图池化比率。（默认值：:obj:`0.5`）
        sum_res (bool, 可选): 如果设置为 :obj:`False`，将使用连接方式整合跳跃连接而不是求和。（默认值：:obj:`True`）
        act (torch.nn.functional, 可选): 使用的非线性函数。（默认值：:obj:`torch.nn.functional.relu`）

    """
class GraphUNet(nn.Module):
        def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        #act: Union[str, Callable] = 'relu',
    ):
        #调用父类的构造函数
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        #self.act = activation_resolver(act)
        self.sum_res = sum_res

        channels = hidden_channels
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        #添加gcn卷积到down conv中
        #第一次的gcn要改变大小
        self.down_convs.append(GCNConv(in_channels, channels))

        for i in range(depth):
            #下采样
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            #之后的gcn不改变大小
            self.down_convs.append(GCNConv(channels, channels))
        #要判断连接方式，根据链接方式来决定下一次的输入通道数
        #如果输出的sum_res为true ，就是求和连接
            if sum_res:
                in_channels = channels
            else:
                in_channels = 2 * channels


        self.up_convs = torch.nn.ModuleList()
        #最下面一层是没有上采样的，所以是-1，那会导致下采样多一次？？
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels))
        self.up_convs.append(GCNConv(in_channels, out_channels))

        self.reset_parameters()
        
    def reset _parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: OptTensor = None) -> Tensor:
        r"""batch不能为空"""
        #创建edge_index的第二维数边的数量num_edges 的一个以为tensor
        edge_weight = x.new_ones(edge_index.size(1))
        #第一个dconv中
        x = self.down_convs[0](x, edge_index, edge_weight)
        #x = self.act(x)
        x = F.relu(x)
        

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            #x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        #移除自环
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        #自己和自己连接
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)

        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        #计算邻接矩阵的平方
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        #最后移除自环
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        #调试用
        """
        obj = MyClass(3, 64, 10, 2, [0.5, 0.25])
        print(repr(obj))
        """
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')