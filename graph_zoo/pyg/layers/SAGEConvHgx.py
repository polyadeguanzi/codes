from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm


class SAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],                       # 输入特征维度
        out_channels: int,                                              # 输出特征维度
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",    # 聚合方式
        normalize: bool = False,                                        # 是否在输出特征上归一化
        root_weight: bool = True,                                       # 是否在计算输出时包括根节点的权重
        bias: bool = True,                                              # 是否在线性变换中使用偏置。
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)                    # 源节点跟目标结点的聚合后的特征维度相同

        super().__init__(aggr, **kwargs)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:                                            # root_weight 参数控制了在计算节点表示时，是否将目标节点自身的特征纳入到聚合过程中
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)                                                  # 源节点和目标节点的特征，为了处理源节点和目标节点的特征可能不同的情况。(源节点，目标节点)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)                # out:(npoint, 1433/128)，只有邻居信息
        out = self.lin_l(out)                                           # (npoint, 128)

        x_r = x[1]                                                      # (npoint, 1433/128)
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)                                 # 特征更新 邻居信息以及中心节点信息

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)                        # p:范数类型

        return out

    def message(self, x_j: Tensor) -> Tensor:
        # print("message")
        return x_j # 邻居结点的信息

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
