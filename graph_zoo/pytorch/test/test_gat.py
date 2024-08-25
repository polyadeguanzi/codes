import torch
from torch import nn
import sys

sys.path.append("/mae_pat/rui.guan/venus-zoo/graph_zoo")
from dataset.read import CoraDataset
from pytorch.models.gat import GAT

def test_gat():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        import torch_gcu
        device = torch_gcu.gcu_device()

    features = CoraDataset().features.to(device)
    adj_mat = CoraDataset().adj_mat.to(device)
    classes = CoraDataset().classes
    model = GAT(
        in_features = features.shape[1], 
        n_hidden = 64,
        n_classes = len(classes), # 节点特征的输出维度
        n_heads = 8, 
        dropout = 0.6
    ).to(device)
    
    pred = model(features, adj_mat)
    assert pred.shape == (features.shape[0], len(classes)) #测试预测结果的形状是否符合预期
    assert pred.isnan().sum() == 0  #测试预测结果数值是否存在nan
    assert pred.isinf().sum() == 0  #测试预测结果数值是否存在inf
    
    sum =  pred.sum()
    sum.backward()                  #测试模型反向过程是否正常运行
    print("finish!")

def accuracy(output: torch.Tensor, labels: torch.Tensor):
    """
    A simple function to calculate the accuracy
    """
    return output.argmax(dim=-1).eq(labels).sum().item() / len(labels)

def run(training_samples=500):
    """
    ### Training loop

    We do full batch training since the dataset is small.
    If we were to sample and train we will have to sample a set of
    nodes for each training step along with the edges that span
    across those selected nodes.
    """
    import csv
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        import torch_gcu
        device = torch_gcu.gcu_device()
    # Move the feature vectors to the device
    features = CoraDataset().features.to(device)
    adj_mat = CoraDataset().adj_mat.to(device)
    classes = CoraDataset().classes
    labels = CoraDataset().labels.to(device)

    # Random indexes
    idx_rand = torch.randperm(len(labels))
    idx_train = idx_rand[:training_samples] # Nodes for training
    idx_valid = idx_rand[training_samples:] # Nodes for validation


    model = GAT(
        in_features = features.shape[1], 
        n_hidden = 64,
        n_classes = len(classes), # 节点特征的输出维度
        n_heads = 8, 
        dropout = 0.6
    ).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    # Training loop
    for epoch in range(1000):
        model.train()
        # Make all the gradients zero
        optimizer.zero_grad()
        # Evaluate the model
        output = model(features, adj_mat)
        loss = loss_func(output[idx_train], labels[idx_train])  # Get the loss for training nodes
        loss.backward()
        optimizer.step()
        # Log the loss
        loss_train = loss.item()
        print('loss_train: ', loss_train)
        accuracy_train = accuracy(output[idx_train], labels[idx_train])
        print('accuracy_train: ', accuracy_train)


        model.eval()
        with torch.no_grad():
            output = model(features, adj_mat)
            loss = loss_func(output[idx_valid], labels[idx_valid])
            # Log the loss
            loss_valid = loss.item()
            print('loss_valid: ', loss_valid)
            accuracy_valid = accuracy(output[idx_valid], labels[idx_valid])
            print('accuracy_valid: ', accuracy_valid)

        # Save loss
        csv_file_path = "/mae_pat/rui.guan/graph_model/train_data/gat_loss.csv"
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([loss_train, accuracy_train, loss_valid, accuracy_valid])

if __name__ == "__main__":
    test_gat()
    # run()
