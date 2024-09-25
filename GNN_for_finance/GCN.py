import pandas as pd
#画图显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import dgl
import numpy as np
from dgl.nn import GraphConv
dat=pd.read_excel('D:/研究生/研一下/统计建模/实证/汽车上市公司向银行借款表/one_mode矩阵2022.xlsx',header=0,index_col=0)
#nx-邻接矩阵创建图
graph = nx.from_numpy_matrix(dat.values)
G=dgl.DGLGraph(graph)
G.ndata['index']=torch.tensor(dat.index)
G= dgl.add_self_loop(G)
feat =pd.read_excel('D:/研究生/研一下/统计建模/实证/汽车上市公司向银行借款表/nodes.xlsx',sheet_name='Sheet5',header=0,index_col=0)
G.ndata['feat'] = torch.tensor(feat.values,dtype=torch.float32)
conv = GraphConv(17, 3, norm='both', weight=True, bias=True) #维降+拓扑降到3维
embed = conv(G, torch.tensor(feat.values,dtype=torch.float32))
conv2 = GraphConv(17, 2, norm='both', weight=True, bias=True)
embed2 = conv2(G, torch.tensor(feat.values,dtype=torch.float32))

dat['embed31']=embed[:,0].detach().numpy()
dat['embed32']=embed[:,1].detach().numpy()
dat['embed33']=embed[:,2].detach().numpy()
dat['embed21']=embed2[:,0].detach().numpy()
dat['embed22']=embed2[:,1].detach().numpy()

from sklearn.ensemble import IsolationForest
ilf = IsolationForest(n_estimators=100,
                          n_jobs=-1,  # 使用全部cpu
                          verbose=2,)

data2=dat[['embed21','embed22']]
    # 训练
ilf.fit(data)
pred=ilf.predict(data)
score=ilf.score_samples(data)
data2['pred2']=pred
data2['score2']=score

df=pd.concat([data,data2],axis=1)
df.to_excel('D:/研究生/研一下/统计建模/实证/网络风险企业识别.xlsx')

'''================画图========================'''
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#matplotlib画图中中文显示会有问题，需要这两行设置默认字体

plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=9,xmin=0)
plt.ylim(ymax=9,ymin=0)
#画两条（0-9）的坐标轴并设置轴标签x，y


colors1 = '#00CED1' #点的颜色
colors2 = '#DC143C'
area = np.pi * 4**2  # 点面积
x1=data2[data2['pred2']==1]['embed21'].values
x2=data2[data2['pred2']==-1]['embed21'].values
y1=data2[data2['pred2']==1]['embed22'].values
y2=data2[data2['pred2']==-1]['embed22'].values

plt.scatter(x1,y1,s=area*3, c=colors1, alpha=0.4, label='正常点')
plt.scatter(x2, y2, s=area*3, c=colors2, alpha=0.4, label='异常点')
plt.legend()
plt.show()

