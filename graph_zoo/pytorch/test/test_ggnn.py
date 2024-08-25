import torch
from torch import nn
import torch.optim as optim
import sys

sys.path.append("/mae_pat/rui.guan/venus-zoo/graph_zoo")
from dataset.babi_data.dataset import bAbIDataset 
from dataset.babi_data.dataset import bAbIDataloader
from pytorch.models.ggnn import GGNN

def test_ggnn():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        import torch_gcu
        device = torch_gcu.gcu_device()

    task_id = 4 # 在数据集中可以为4 15 16
    dataroot='/mae_pat/rui.guan/venus-zoo/graph_zoo/dataset/babi_data/processed_1/train/%d_graphs.txt' % task_id
    batchSize=10

    train_dataset = bAbIDataset(dataroot, True)
    train_dataloader = bAbIDataloader(train_dataset, batch_size=batchSize, \
                                      shuffle=True, num_workers=2)
    test_dataset = bAbIDataset(dataroot,False)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=batchSize, \
                                     shuffle=False, num_workers=2)
    
    # for bAbI
    state_dim = 4   # GGNN hidden state dim
    annotation_dim = 1  
    n_edge_types = train_dataset.n_edge_types
    n_node = train_dataset.n_node
    n_steps = 5 # propogation steps number of GGNN

    net = GGNN(state_dim,
                annotation_dim,
                n_edge_types,
                n_node,
                n_steps
                )
    # net.float()
    print(net)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        net.to(device)
        criterion.to(device)

    lr = 0.01
    optimizer = optim.Adam(net.parameters(), lr=lr)

    niter = 10  # number of epochs to train for
    for epoch in range(0, niter):
        # train
        net.train()
        for i, (adj_matrix, annotation, target) in enumerate(train_dataloader, 0):
            adj_matrix = adj_matrix.type(torch.float32)
            annotation = annotation.type(torch.float32)
            # target = target.type(torch.float32)
            net.zero_grad()
            # 将注释特征填充到指定维度state_dim
            padding = torch.zeros(len(annotation), n_node, state_dim - annotation_dim, dtype=torch.float32)
        # opt.state_dim为hidden state dim， opt.annotation_dim为标注的dim
            init_input = torch.cat((annotation, padding), 2)
            init_input = init_input.to(device)
            adj_matrix = adj_matrix.to(device)
            annotation = annotation.to(device)
            target = target.to(device)

            output = net(init_input, annotation, adj_matrix)    # TODO: 解释输入含义
            assert output.isnan().sum() == 0  #测试预测结果数值是否存在nan
            assert output.isinf().sum() == 0  #测试预测结果数值是否存在inf
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            # 打印日志
            if i % int(len(train_dataloader) / 10 + 1) == 0 :
                print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, niter, i, len(train_dataloader), loss.item()))

        # eval
        test_loss = 0
        correct = 0
        net.eval()
        for i, (adj_matrix, annotation, target) in enumerate(test_dataloader, 0):
            adj_matrix = adj_matrix.type(torch.float32)
            annotation = annotation.type(torch.float32)
            padding = torch.zeros(len(annotation), n_node, state_dim - annotation_dim, dtype=torch.float32)
            init_input = torch.cat((annotation, padding), 2)
            init_input = init_input.to(device)
            adj_matrix = adj_matrix.to(device)
            annotation = annotation.to(device)
            target = target.to(device)

            output = net(init_input, annotation, adj_matrix)
            print(output)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_dataloader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))




if __name__ == "__main__":
    test_ggnn()