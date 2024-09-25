import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.utils import negative_sampling
from torch_geometric.nn.inits import reset
from typing import Optional, Tuple

EPS = 1e-15

class InnerProductDecoder(torch.nn.Module):
    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self,data) -> Tensor:
        r"""Encodes the input data and then decodes it to produce edge probabilities."""
        z = self.encode(data)
        return self.decode(z, data.edge_index)

    def encode(self, data) -> Tensor:
        r"""Encodes the input data to produce the latent space :obj:`z`."""
        return self.encoder(data)

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities."""
        return self.decoder(z, edge_index)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:

        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
