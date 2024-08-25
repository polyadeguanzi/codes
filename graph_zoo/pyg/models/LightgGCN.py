from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss
import sys
sys.path.append('/mae_pat/xi.sun/venus-zoo/graph_zoo')
#from torch_geometric.nn.conv import LGConv
from pyg.layers.LGConv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index


class LightGCN(torch.nn.Module):

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes  #节点数量
        self.embedding_dim = embedding_dim #embedding维数
        self.num_layers = num_layers #迭代多少次图卷积

        if alpha is None:
            alpha = 1. / (num_layers + 1) #alpaha用于存储每层的normalization系数，这里是在初始化

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])#定义卷积层

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight) #迭代地进行图卷积
            out = out + x * self.alpha[i + 1]

        return out

    def forward(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Computes rankings for pairs of nodes.
        """
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)#更新后的节点embedding

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]
        #sum(逐元素相乘的结果)
        return (out_src * out_dst).sum(dim=-1)

    #predict_link与recommend都是下游任务
    def predict_link(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        prob: bool = False,
    ) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.
        链接预测，返回连边的概率
        """
        pred = self(edge_index, edge_label_index, edge_weight).sigmoid()
        return pred if prob else pred.round()

    def recommend(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
        sorted: bool = True,
    ) -> Tensor:
        r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`.
        推荐，返回topk个推荐的index
        """
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1, sorted=sorted).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.
        计算链接预测损失
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def recommendation_loss(
        self,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        node_id: Optional[Tensor] = None,
        lambda_reg: float = 1e-4,
        **kwargs,
    ) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss.
        计算推荐损失
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        emb = self.embedding.weight
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')


class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.
    计算推荐损失
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.
        """
        log_prob = F.logsigmoid(positives - negatives).mean()

        regularization = 0
        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2) #lambda_reg 用于控制嵌入参数的正则化强度。正则化在机器学习模型中常用于防止过拟合
            regularization = regularization / positives.size(0)

        return -log_prob + regularization