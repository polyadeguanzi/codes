import torch.nn as nn
import math
import torch
import random
import pyhocon
import argparse
import numpy as np
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
class SageLayer(nn.Module):
    def __init__(self, input_size, out_size, gcn=False): 
        super(SageLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size)) # 创建weight
        self.init_params()                                                # 初始化参数

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)   # concat自己信息和邻居信息
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class GraphSage(nn.Module):

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        #设置参数以及邻接矩阵
        self.input_size = input_size                                  # 输入尺寸   1433
        self.out_size = out_size                                      # 输出尺寸   128
        self.num_layers = num_layers                                  # 聚合层数   2
        self.gcn = gcn                                                # 是否使用GCN
        self.device = device                                          # 使用训练设备
        self.agg_func = agg_func                                      # 聚合函数
        self.raw_features = raw_features                              # 节点特征
        self.adj_lists = adj_lists                                    # 边
        
        #设置sage_layer的属性
        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size       # 如果index==1，这中间特征为1433，如果！=1。则特征数为128。
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))  

    def forward(self, nodes_batch):

        #Step 1 加载node_batch，node_batch是从输入的整张图中随机选取的20个点；每个批次以这20个点为中心进行采样和聚合
        lower_layer_nodes = list(nodes_batch)                          # 把当前训练的节点转换成list
        # [527, 1681, 439, 2007, 1439, 963, 699, 131, 1003, 1, 658, 1660, 16, 716, 245, 2577, 501, 1582, 1081, 944] dim=20=2*10=k*s
        nodes_batch_layers = [(lower_layer_nodes,)]                    # 放入的训练节点
        # [([527, 1681, 439, 2007, 1439, 963, 699, 131, 1003, 1, 658, 1660, 16, 716, 245, 2577, 501, 1582, 1081, 944],)]
        
        #Sep2 从1到k=num_layer，采样
        for i in range(self.num_layers):                               # 遍历每一次聚合，获得neighbors
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)  
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
                                     # batch涉及到的所有节点，本身+邻居 ，      节点编号->当前字典中顺序index    
            #[([涉及到的所有节点],[{邻居+自己},{邻居+自己}],{节点index}),([batch节点]),] 
        assert len(nodes_batch_layers) == self.num_layers + 1
        pre_hidden_embs = self.raw_features #初始化节点的embedding

        #Step3 从k=num_layer到1，聚合自己和邻居的节点
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]                           #自己和邻居点
            pre_neighs = nodes_batch_layers[index-1]                    # 第num_layers+1-index层的节点编号（范围0-2077），自己和邻居节点，邻居节点编号->字典中编号
            #第num_layers+1-index层和num_layers-index层的聚合（例如,num_layers=2,index=1，即是2跳邻居聚合将消息传给1跳邻居）
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)   # 聚合函数。聚合的节点， 节点特征，集合节点邻居信息
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)   # 第一层的batch节点，没有进行转换
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb], aggregate_feats=aggregate_feats)  
                                                                        # 进入SageLayer。weight*concat(node,neighbors)
            pre_hidden_embs = cur_hidden_embs
        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]                    # 记录将上一层的节点编号。
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        #单层/对于某个特定的深度的采样函数
        
        #Step 1: 从原始的邻接矩阵获取每个节点的邻居id
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]       # self.adj_lists边矩阵，获取节点的邻居
        
        #Step2 对邻居节点进行采样，如果大于邻居数据，则进行采样
        if not num_sample is None:                                      
            _sample = random.sample                                     # 节点长度小于10 
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        
        #Step3 存储采样后的邻居，以及新生成的连续编号
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # 每个都加入本身节点
        _unique_nodes_list = list(set.union(*samp_neighs))               # 这个batch涉及到的所有节点
        i = list(range(len(_unique_nodes_list)))                         # 建立这个batch新的节点编号
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))            # 节点编号->当前字典中顺序index
        return samp_neighs, unique_nodes, _unique_nodes_list             # 聚合自己和邻居节点，点的dict，batch涉及到的所有节点

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        #聚合函数

        #加载sampled数据
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs        # batch涉及到的所有节点,本身+邻居,邻居节点编号->字典中编号  
        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]  # 是否包含本身
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]  # 在把中心节点去掉
        if len(pre_hidden_embs) == len(unique_nodes):                     # 保留需要使用的节点特征。
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]                                               
        
        #生成小的（sampled）邻接矩阵与特征矩阵
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))           # (本层节点数量，邻居节点数量)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]  # 保存列 每一行对应的邻居真实index做为列。
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]# 保存行 每行邻居数
        mask[row_indices, column_indices] = 1                             # 构建邻接矩阵;
        
        #不同聚合类型进行不同的聚合运算
        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)                         # 按行求和，保持和输入一个维度
            mask = mask.div(num_neigh).to(embed_matrix.device)            # 归一化
            aggregate_feats = mask.mm(embed_matrix)                       # 矩阵相乘，相当于聚合周围邻接信息求和
        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask==1]                       # x用于记录元素是否非0，index记录非零元素的位置
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:      #取出邻居的emb;如果只有一个邻居，用改邻居emb更新自己的；如果有多个，选最大值
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)
        return aggregate_feats

