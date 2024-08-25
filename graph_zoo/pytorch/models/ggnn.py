import torch
import torch.nn as nn
from pytorch.layers.attention import gru

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, 
                state_dim,
                annotation_dim,
                n_edge_types,
                n_node,
                n_steps
                ):
        super(GGNN, self).__init__()

        assert (state_dim >= annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_node = n_node
        self.n_steps = n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim, dtype=torch.float32)
            out_fc = nn.Linear(self.state_dim, self.state_dim, dtype=torch.float32)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = gru(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        # prop_state:论文中的h
        # annotation:节点标注，每个节点的额外信息或特征。
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            # 对输入的边，按类型进行线性变换
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)   # 边的输入端节点特征
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim) # 边的输出端节点特征

            prop_state = self.propogator(in_states, out_states, prop_state, A)  # 消息传播

        join_state = torch.cat((prop_state, annotation), 2)

        output = self.out(join_state)
        output = output.sum(2)

        return output
