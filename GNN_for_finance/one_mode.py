import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import networkx as nx
#画图显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
'''构建金融网络'''
####
dat=pd.read_excel('D:/研究生/研一下/统计建模/实证/汽车上市公司向银行借款表/CBL_LoanNew.xlsx',sheet_name='Sheet5',header=0,index_col=None)
firm=set(list(dat['Symbol']))
df=pd.DataFrame(np.zeros((55,55)),index=firm,columns=firm)
for i in firm:
    banki=set(dat[dat['Symbol']==i]['LoanBank'])
    for j in firm:
        if i!=j:
            for k in dat[dat['Symbol']==j]['LoanBank']:
                if k in banki:
                    df[i][j]+=1

df.to_excel('D:/研究生/研一下/统计建模/实证/one_mode矩阵2022.xlsx')
graph = nx.from_numpy_matrix(df.values)

nx.write_gexf(graph,'D:/研究生/研一下/统计建模/实证/网络2022.gexf')