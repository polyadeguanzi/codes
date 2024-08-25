import torch.nn as nn
import torch
class Classification(nn.Module):                                         
    '''把GraphSAGE的输出链接全连接层每个节点映射到7维+softmax'''
    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()
        self.layer = nn.Sequential(nn.Linear(emb_size, num_classes))      
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        logists = torch.log_softmax(self.layer(embeds), 1)
        return logists