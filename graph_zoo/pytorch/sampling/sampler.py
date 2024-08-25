import pdb
import math
import torch
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import norm as sparse_norm
from torch.nn.parameter import Parameter

import sys
sys.path.append("/mae_pat/rui.guan/venus-zoo/graph_zoo")
from pytorch.sampling.utils import sparse_mx_to_torch_sparse_tensor, load_data


class Sampler:
    def __init__(self, features, adj, **kwargs):
        allowed_kwargs = {'input_dim', 'layer_sizes', 'device'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, \
                'Invalid keyword argument: ' + kwarg

        self.input_dim = kwargs.get('input_dim', 1)
        self.layer_sizes = kwargs.get('layer_sizes', [1])
        self.scope = kwargs.get('scope', 'test_graph')
        self.device = kwargs.get('device', torch.device("cpu"))

        self.num_layers = len(self.layer_sizes)

        self.adj = adj
        self.features = features

        self.train_nodes_number = self.adj.shape[0]

    def sampling(self, v_indices):
        raise NotImplementedError("sampling is not implimented")

    def _change_sparse_to_tensor(self, adjs):
        new_adjs = []
        for adj in adjs:
            new_adjs.append(
                sparse_mx_to_torch_sparse_tensor(adj).to(self.device))
        return new_adjs


class Sampler_FastGCN(Sampler):
    def __init__(self, pre_probs, features, adj, **kwargs):
        super().__init__(features, adj, **kwargs)
        # NOTE: uniform sampling can also has the same performance!!!!
        # try, with the change: col_norm = np.ones(features.shape[0])
        col_norm = sparse_norm(adj, axis=0) # 均匀采样
        self.probs = col_norm / np.sum(col_norm)    # TODO: 为什么没有按公式做乘方运算

    def sampling(self, v):
        """
        Inputs:
            v: batch nodes list
        """
        all_support = [[]] * self.num_layers    # 存放每一层的邻接矩阵

        cur_out_nodes = v   # 节点索引,[256]
        # 从最后一层开始（最深的层），向前遍历每一层。
        for layer_index in range(self.num_layers-1, -1, -1):
            cur_sampled, cur_support = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_support[layer_index] = cur_support  # 得到每一层的采样的邻接矩阵
            cur_out_nodes = cur_sampled 

        all_support = self._change_sparse_to_tensor(all_support)    # 采样后的节点的邻接矩阵
        sampled_X0 = self.features[cur_out_nodes]   # 采样后的节点列表
        return sampled_X0, all_support, 0

    def _one_layer_sampling(self, v_indices, output_size):
        # NOTE: FastGCN described in paper samples neighboors without reference
        # to the v_indices. But in its tensorflow implementation, it has used
        # the v_indice to filter out the disconnected nodes. So the same thing
        # has been done here.
        
        # 1、取每个节点的邻接节点下标
        support = self.adj[v_indices, :]    # 下标v_indices对应节点有哪些邻接节点
        neis = np.nonzero(np.sum(support, axis=0))[1]   # 下标v_indices对应的邻接节点,[558] #TODO: 为什么通过sum的方式取邻接节点

        # 2、计算每个邻居节点的采样概率
        p1 = self.probs[neis]   # [558]
        p1 = p1 / np.sum(p1)    # [558]

        # 3、根据计算的概率分布从邻居节点中进行采样
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   output_size, True, p1)   # 得到下标

        # 4、对应被采样的节点及其邻接矩阵
        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))   # TODO: 解释该行的作用
        return u_sampled, support # 采样后的邻居节点索引，采样后的邻接矩阵


class Sampler_ASGCN(Sampler, torch.nn.Module):
    def __init__(self, pre_probs, features, adj, **kwargs):
        super().__init__(features, adj, **kwargs)
        torch.nn.Module.__init__(self)
        self.feats_dim = features.shape[1]

        # attention weights w1 is also wg
        self.w1 = Parameter(torch.FloatTensor(self.feats_dim, 1))
        self.w2 = Parameter(torch.FloatTensor(self.feats_dim, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.w1.size(0))
        self.w1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-stdv, stdv)

    def sampling(self, v):
        """
        Inputs:
            v: batch nodes list
        """
        v = torch.LongTensor(v)
        all_support = [[]] * self.num_layers    # 存放每一层的邻接矩阵
        all_p_u = [[]] * self.num_layers    # 存放每一层计算方差所需要的参数,包含采样前的邻接节点下标和对应的采样概率

        # sample top-1 layer
        # all_x_u[self.num_layers - 1] = self.features[v]
        cur_out_nodes = v
        for i in range(self.num_layers-1, -1, -1):
            cur_u_sampled, cur_support, cur_var_need = \
                self._one_layer_sampling(cur_out_nodes,
                                         output_size=self.layer_sizes[i])

            all_support[i] = cur_support
            all_p_u[i] = cur_var_need
            cur_out_nodes = cur_u_sampled

        loss = self._calc_variance(all_p_u)
        sampled_X0 = self.features[cur_out_nodes]
        return sampled_X0, all_support, loss

    def _calc_variance(self, var_need):
        # NOTE: it's useless in this implementation for the three datasets
        # only calc the variane of the last layer
        u_nodes, p_u = var_need[-1][0], var_need[-1][1] # -1表示取最后一层
        p_u = p_u.reshape(-1, 1)
        feature = self.features[u_nodes]
        means = torch.sum(feature, 0)
        feature = feature - means
        var = torch.mean(torch.sum(torch.mul(feature, feature) * p_u, 0))
        return var

    def _one_layer_sampling(self, v_indices, output_size):
        # 1、取每个节点的邻接节点下标
        support = self.adj[v_indices, :]    # 下标v_indices对应节点有哪些邻接节点
        neis = np.nonzero(np.sum(support, axis=0))[1]  # 下标v_indices对应的邻接节点索引,[558]  #TODO: 为什么通过sum的方式取邻接节点
        support = support[:, neis]   #  采样前的邻接矩阵,[256*558]
        # NOTE: change the sparse support to dense, mind the matrix size
        support = support.todense()
        support = torch.FloatTensor(support).to(self.device)
        h_v = self.features[v_indices]  # 获取输入节点特征
        h_u = self.features[neis]   # 获取邻居节点特征

        # 2、计算每个邻居节点的采样概率
        attention = torch.mm(h_v, self.w1) + \
            torch.mm(h_u, self.w2).reshape(1, -1) + 1
        attention = (1.0 / np.size(neis)) * torch.relu(attention)   # 对注意力权重进行归一化,[256*558]

        p1 = torch.sum(support * attention, 0)  # 邻居节点的采样概率,[558]  #TODO: 确认p1的作用
        # sampling only done in CPU
        numpy_p1 = p1.to('cpu').data.numpy()
        numpy_p1 = numpy_p1 / np.sum(numpy_p1)

        # 3、根据计算的概率分布从邻居节点中进行采样
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   size=output_size,
                                   replace=True,
                                   p=numpy_p1)  # sampled,[128]

        # 4、对应被采样的节点及其邻接矩阵
        u_sampled = neis[sampled]
        support = support[:, sampled]
        sampled_p1 = p1[sampled]

        t_diag = torch.diag(1.0 / (sampled_p1 * output_size))   # TODO: 解释该行的作用
        support = torch.mm(support, t_diag)

        return u_sampled, support, (neis, p1 / torch.sum(p1))
    # 采样后的邻居节点索引 u_sampled、采样后的邻接矩阵 support，以及一个元组 (neis, p1 / torch.sum(p1))，其中包含邻居节点索引和归一化后的采样概率。


if __name__ == '__main__':
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        load_data("cora")
    batchsize = 256
    layer_sizes = [128, 128, batchsize]
    input_dim = features.shape[1]

    sampler = Sampler_ASGCN(None, train_features, adj_train,
                            input_dim=input_dim,
                            layer_sizes=layer_sizes, scope="None")

    batch_inds = list(range(batchsize))
    sampled_feats, sampled_adjs, var_loss = sampler.sampling(batch_inds)
    pdb.set_trace()
