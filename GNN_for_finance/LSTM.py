import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#画图显示中文
from pylab import mpl
from sklearn import metrics
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
####
dat=pd.read_excel('D:/研究生/研一下/统计建模/实证/汇总数据填充.xlsx',sheet_name='Sheet1',header=0,index_col=None)
dat['统计截止日期']=pd.to_datetime(dat['统计截止日期'], format='%Y-%m-%d')
dat.index=dat['统计截止日期']
I=list(set(dat['股票代码']))
col_index=list(dat.columns)
columns_to_resample = col_index[6:-4]+[col_index[-1]]
d={}
e={'MSE':0,'RMSE':0,'MAE':0,'MAPE':0}
for ind in I:
    print(ind)
    #data =dat[dat['股票代码']==ind].loc[:,columns_to_resample].apply(lambda x: x.resample('M').mean())
    #data = data.fillna(data.bfill())
    #不让最后一期作为训练集
    data = dat[dat['股票代码']==ind][columns_to_resample].iloc[:-4,]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data[columns_to_resample] .values.reshape(-1, 17))
    x=[]
    y=[]
    time_back=1
    for i in range(len(data)-time_back-1):
        x.append(data[i:i+time_back])
        y.append(data[i+time_back])
    x=np.array(x)
    y=np.array(y)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.3)

    #现在，我们可以将数据转换为张量，并将其移到GPU上（如果有的话）：
    train_X = torch.from_numpy(train_X).float()
    train_Y = torch.from_numpy(train_Y).float()
    test_X = torch.from_numpy(test_X).float()
    test_Y = torch.from_numpy(test_Y).float()

    # 定义神经网络模型
    class LSTM(nn.Module):
        def __init__(self, input_size, output_size):
            super(LSTM, self).__init__()

            # 构建隐藏层
            self.fc1 = nn.LSTM(input_size, 100)
            self.fc2 = nn.LSTM(100, 200)
            self.fc3 = nn.Linear(200, output_size)

        def forward(self, x):
            x = self.fc1(x)[0]
            x = self.fc2(x)[0]
            x = self.fc3(x)
            return x


    # 定义数据和超参数
    input_size = 17
    output_size = 17
    learning_rate = 0.01
    num_epochs = 280



    # 初始化神经网络模型、损失函数和优化器
    net = LSTM(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 训练神经网络
    for epoch in range(num_epochs):
        # 前向传播
        output = net(train_X.reshape(-1, input_size).clone().detach())
        loss = criterion(output, train_Y.reshape(-1, input_size).clone().detach())
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        if epoch % 50 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    '''--------------（样本外）预测---------------------'''
    pred_X=data[-1]
    pred=net(torch.tensor(pred_X.reshape(-1, input_size), dtype=torch.float32))
    pred_inv= scaler.inverse_transform(pred.detach().numpy())
    d[ind]=pred

    e['MSE'] = e['MSE']+metrics.mean_squared_error(pred_X, pred.detach().numpy())
    e['RMSE'] =e['RMSE']+ metrics.mean_squared_error(pred_X, pred.detach().numpy()) ** 0.5
    e['MAE'] =e['MAE']+ metrics.mean_absolute_error(pred_X, pred.detach().numpy())
    #e['MAPE']= e['MAPE']+metrics.mean_absolute_percentage_error(pred_X, pred.detach().numpy()).mean()

df = pd.DataFrame(np.zeros((79*5,18)),columns=columns_to_resample.append('股票代号'))
for i in range(79*5):
            df.iloc[i,17]=list(d.keys())[i//5]
            df.iloc[i,:17]=list(d[list(d.keys())[i//5]][i%5])

df.to_excel('D:/研究生/研一下/统计建模/实证/202201_202301删除版.xlsx')
#pred_X取data[-2]表示用2022年第三季度预测出第四季度

