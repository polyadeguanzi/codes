import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from imblearn.over_sampling import SMOTE

data=pd.read_excel('D:/研究生/研一下/统计建模/实证/删除版.xlsx',sheet_name='Sheet4',header=0,index_col=0)
data['risk']=round(data['risk'])
x_cols=pd.read_excel('D:/研究生/研一下/统计建模/实证/删除版.xlsx',sheet_name='Sheet4',header=None,index_col=None).iloc[0,5:]
x_cols=list(x_cols)

l=data
ilf = IsolationForest(n_estimators=100,
                          n_jobs=-1,  # 使用全部cpu
                          verbose=2,)

    # 训练
ilf.fit(l[x_cols])
pred=ilf.predict(l[x_cols])
score=ilf.score_samples(l[x_cols])
data['pred']=pred
data['score']=score

up=data[data['上下游']=='上游'].iloc[:,4:]
down=data[data['上下游']=='下游'].iloc[:,4:]
mid=data[data['上下游']=='中游'].iloc[:,4:]


ls=[up,mid,down]
pred=[]
score=[]
for l in ls:
    l = (l - l.min()) / (l.max() - l.min())
    ilf = IsolationForest(n_estimators=100,
                          n_jobs=-1,  # 使用全部cpu
                          verbose=2,)

    # 训练
    ilf.fit(l)
    pred.append(ilf.predict(l))
    score.append(ilf.score_samples(l))


up['pred']=pred[0]
mid['pred']=pred[1]
down['pred']=pred[2]
up['score']=score[0]
mid['score']=score[1]
down['score']=score[2]
df=pd.concat((up,mid,down),axis=0)

data=data.join(df[['pred','score']])
data.to_excel('D:/研究生/研一下/统计建模/实证/孤立森林202104不分.xlsx')

'''=============样本外预测====================='''
data=pd.read_excel('D:/研究生/研一下/统计建模/实证/202301预测截面不逆.xlsx',sheet_name='Sheet1',header=0,index_col=0)

ilf = IsolationForest(n_estimators=100,
                          n_jobs=-1,  # 使用全部cpu
                          verbose=2,)

    # 训练
ilf.fit(data)
pred=ilf.predict(data)
score=ilf.score_samples(data)
data['pred']=pred
data['score']=score

data.to_excel('D:/研究生/研一下/统计建模/实证/202301风险企业识别2.xlsx')
