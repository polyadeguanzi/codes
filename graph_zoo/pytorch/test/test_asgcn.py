import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys
sys.path.append("/mae_pat/rui.guan/venus-zoo/graph_zoo")
from pytorch.models.gcn import GCN
from pytorch.sampling.sampler import Sampler_FastGCN, Sampler_ASGCN
from pytorch.sampling.utils import load_data, get_batches, accuracy
from pytorch.sampling.utils import sparse_mx_to_torch_sparse_tensor


def train(model, optimizer, loss_fn, train_ind, train_labels, batch_size,
                                                  nclass, train_times):
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches(train_ind,
                                                    train_labels,
                                                    batch_size):
            sampled_feats, sampled_adjs, var_loss = model.sampling(
                batch_inds)
            optimizer.zero_grad()
            assert sampled_feats.shape == (128, 1433)
            output = model(sampled_feats, sampled_adjs)
            # assert output.shape == (batch_size, nclass) #测试预测结果的形状是否符合预期
            print(sampled_feats.shape[0],output.shape)
            assert output.isnan().sum() == 0  #测试预测结果数值是否存在nan
            assert output.isinf().sum() == 0  #测试预测结果数值是否存在inf
            loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t


def test(model, optimizer, loss_fn, test_adj, test_feats, test_labels, epoch):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_adj)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t


def test_asgcn():
    # 参数设置
    dataset = 'cora'
    seed = 123
    epochs = 100
    batchsize = 256 # 每次采样取256个节点
    # load data, set superpara and constant
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        load_data(dataset)

    layer_sizes = [128, 128]
    input_dim = features.shape[1]
    train_nums = adj_train.shape[0]
    test_gap = 10 # the train epochs between two test
    nclass = y_train.shape[1]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        import torch_gcu
        device = torch_gcu.gcu_device()

    # data for train and test
    features = torch.FloatTensor(features).to(device)
    train_features = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(y_train).to(device).max(1)[1]

    test_adj = [adj, adj[test_index, :]]
    test_feats = features
    test_labels = y_test
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                for cur_adj in test_adj]
    test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]

    # init the sampler
    sampler = Sampler_ASGCN(None, train_features, adj_train,
                                input_dim=input_dim,
                                layer_sizes=layer_sizes,
                                device=device)

    # init model, optimizer and loss function
    model = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=nclass,
                dropout=0.0,
                sampler=sampler).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.01, weight_decay=5e-4)
    loss_fn = F.nll_loss

    # train and test
    for epochs in range(0, epochs // test_gap):
        train_loss, train_acc, train_time = train(model, optimizer, loss_fn, 
                                                  np.arange(train_nums),
                                                  y_train,
                                                  batchsize,
                                                  nclass,
                                                  test_gap)
        test_loss, test_acc, test_time = test(model, optimizer, loss_fn, 
                                              test_adj,
                                              test_feats,
                                              test_labels,
                                              epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s")

if __name__ == '__main__':
    test_asgcn()