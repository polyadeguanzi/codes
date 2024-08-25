import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    """
    ## Graph attention layer

    This is a single graph attention layer.
    A GAT is made up of multiple such layers.

    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
    """
    def __init__(self, in_features: int, 
                 out_features: int, 
                 n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * h: the input node embeddings of shape [n_nodes, in_features]
        * adj_mat: the adjacency matrix of shape [n_nodes, n_nodes, n_heads]
        We use shape [n_nodes, n_nodes, 1] since the adjacency is the same for each head.
        """
        n_nodes = h.shape[0]    # Number of nodes

        # 1、节点特征的预处理
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)   # 节点×共享参数     [n_nodes, n_heads, n_hidden]    将in_features按注意力头数拆分为n_heads * n_hidden
        g_repeat = g.repeat(n_nodes, 1, 1)  # 所有节点重复n_node次  node1, node2, ..., noden, node1, node2, ..., noden,...    [n_nodes * n_nodes, n_heads, n_hidden]
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)   # 每个节点重复n_node次  node1, node1, ..., node1, node2, node2, ..., node2,...    [n_nodes * n_nodes, n_heads, n_hidden]

        # 2、拼接中心节点及其邻接节点特征
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)   # 拼接节点与相邻节点    [n_nodes * n_nodes, n_heads, 2*n_hidden]
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden) # 每个节点与其他节点进行增维操作    [n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden]

        # 3、计算注意力权重
        e = self.activation(self.attn(g_concat))    # 计算注意力分数，并对其归一化取softmax      [n_nodes, n_nodes, self.n_heads, 1]
        e = e.squeeze(-1)   # [n_nodes, n_nodes, self.n_heads]
        # 注意：每个节点与其他所有节点（包括实际的邻接节点和非邻接节点）都进行了注意力计算

        adj_mat = adj_mat.unsqueeze(-1).repeat(1, 1, self.n_heads)
        e = e.masked_fill(adj_mat == 0, float('-inf'))  # 将邻接矩阵中的0设置为-inf，这是为了取sotfmax后，邻接矩阵中的0仍然为0，每个节点只与相邻节点计算注意力权重，保证了只获取相邻节点的信息
        a = self.softmax(e) # 按邻接矩阵进行mask, 取激活函数
        a = self.dropout(a) # 得到注意力权重    [n_nodes, n_nodes, self.n_heads]
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)   # 注意力权重×节点×共享参数  [n_nodes, self.n_heads, self.n_hidden]
        # i = n_nodes, j = n_nodes, h = self.n_heads, f = self.n_hidden

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)  # [n_nodes, self.n_heads * self.n_hidden]
        else:
            return attn_res.mean(dim=1)

class gru(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(gru, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        """
            state_in: 输入边节点特征
            state_out: 输出边节点特征
            state_cur：当前节点特征
            A: [A_in,A_out]，A_in当前节点的边输入关系，A_out当前节点的边输出关系
        """
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)
        
        # GRU
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output