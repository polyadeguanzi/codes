import torch
import torch.nn as nn
#from torch_geometric.nn import PNAConv
import sys
sys.path.append('/mae_pat/xi.sun/venus-zoo')
from graph_zoo.pyg.layers.PNAConv import PNAConv
class PNA(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None,
                 dropout=0.0, act='relu', act_first=False, act_kwargs=None,
                 norm=None, norm_kwargs=None, jk=None, **kwargs):
        super(PNA, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize PNAConv layers
        self.conv_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                conv = PNAConv(in_channels, hidden_channels, **kwargs)
            elif layer == num_layers - 1 and out_channels is not None:
                conv = PNAConv(hidden_channels, out_channels, **kwargs)
            else:
                conv = PNAConv(hidden_channels, hidden_channels, **kwargs)
            self.conv_layers.append(conv)

        # Optional final linear transformation for jumping knowledge
        if jk is not None:
            if jk == 'cat':
                self.jk_transform = nn.Linear(num_layers * hidden_channels, out_channels)
            elif jk in ['last', 'max', 'lstm']:
                self.jk_transform = nn.Linear(hidden_channels, out_channels)
            else:
                raise ValueError(f"Unsupported JK mode '{jk}'")
        else:
            self.jk_transform = None

        # Activation function
        self.activation = nn.ReLU() if act == 'relu' else act

        # Dropout and normalization
        self.dropout_layer = nn.Dropout(p=dropout)
        self.norm_layer = nn.BatchNorm1d(hidden_channels) if norm == 'batch' else None

    def forward(self, x, edge_index):
        for layer in range(self.num_layers):
            x = self.conv_layers[layer](x, edge_index)

            # Apply activation and normalization
            if self.norm_layer is not None:
                x = self.norm_layer(x)
            if layer < self.num_layers - 1:  # apply dropout except for the last layer
                x = self.activation(x)
                x = self.dropout_layer(x)

        # Apply jumping knowledge if specified
        if self.jk_transform is not None:
            if jk == 'cat':
                x = torch.cat([x[layer] for layer in range(self.num_layers)], dim=-1)
            elif jk == 'last':
                x = x[-1]
            elif jk == 'max':
                x, _ = x.max(dim=0)
            elif jk == 'lstm':
                x = self.jk_lstm(x)
            x = self.jk_transform(x)

        return x
